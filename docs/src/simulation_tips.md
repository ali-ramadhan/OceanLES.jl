
# Simulation tips

In Oceananigans we try to do most of the optimizing behind the scenes, that way the average user
doesn't have to worry about details when setting up a simulation. However, there's just so much
optimization that can be done in the source code. Because of Oceananigans script-based interface,
the user has to be aware of some things when writing the simulation script in order to take full
advantage of Julia's speed. Furthermore, in case of more complex GPU runs, some details could
sometimes prevent your simulation from running altogether. While Julia knowledge is obviously
desirable here, a user that is unfamiliar with Julia can get away with efficient simulations by
learning a few rules of thumb. It is nonetheless recommended that users go through Julia's
[performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/), which contains more
in-depth explanations of some of the aspects discussed here.



## General (CPU/GPU) simulation tips

### Avoid global variables whenever possible

In general using a [global
variable](https://docs.julialang.org/en/v1/manual/variables-and-scoping/#Global-Scope) (which can be
loosely defined as a variable defined in the main script) inside functions slows down the code. One
way to circumvent this is to always [use local variables or pass them as arguments to
functions](https://docs.julialang.org/en/v1/manual/performance-tips/#Avoid-global-variables). This
helps the compiler optimize the code.

Another way around this is to [define global variables as constants whenever
possible](https://docs.julialang.org/en/v1/manual/performance-tips/#Avoid-global-variables). One
thing to keep in mind when doing this is that when a `const` is defined, its value can't be changed
until you restart the Julia session. So this latter approach is good for production-ready code, but
may be undesirable in the early stages of development while you still have to change the parameters
of the simulation for exploration.

It is especially important to avoid global variables in functions that are meant to be executed in
GPU kernels (such as functions defining boundary conditions and forcings). Otherwise the Julia GPU
compiler can fail with obscure errors. This is explained in more detail in the GPU simulation tips
section below.


### Consider inlining small functions

Inlining is when the compiler [replaces a function call with the body of the function that is being
called before compiling](https://en.wikipedia.org/wiki/Inline_expansion). The advantage of inlining
(which in julia can be done with the [`@inline`
macro](https://docs.julialang.org/en/v1/devdocs/meta/)) is that gets rid of the time spent calling
the function. The Julia compiler automatically makes some calls as to what functions it should or
shouldn't inline, but you can force a function to be inlined by including the macro `@inline` before
its definition. This is more suited for small functions that are called often. Here's an example of
an implementation of the Heaviside function that forces it to be inlined:

```julia
@inline heaviside(X) = ifelse(X < 0, zero(X), one(X))
```

In practice it's hard to say whether inlining a function will bring runtime benefits _with
certainty_, since Julia and KernelAbstractions.jl (needed for GPU runs) already inline some
functions automatically. However, it is generally a good idea to at least investigate this aspect in
your code as the benefits can potentially be significant.



## GPU simulation tips

Running on GPUs can be very different from running on CPUs. Oceananigans makes most of the necessary
changes in the background, so that for very simple simulations changing between CPUs and GPUs is
just a matter of changing the `architecture` argument in the model from `CPU()` to `GPU()`. However,
for more complex simulations some care needs to be taken on the part of the user. While knowledge of
GPU computing (and Julia) is again desirable, an inexperienced user can also achieve high efficiency
in GPU simulations by following a few simple principles.


### Variables that need to be used in GPU computations need to be defined as constants

Any global variable that needs to be accessed by the GPU needs to be a constant or the simulation
will crash. This includes any variables that are referenced as global variables in functions
used for forcing of boundary conditions. For example,

```julia
T₀ = 20 # ᵒC
surface_temperature(x, y, t) = T₀ * sin(2π / 86400 * t)
T_bcs = FieldBoundaryConditions(bottom = GradientBoundaryCondition(surface_temperature))
```

will throw an error if run on the GPU (and will run more slowly than it should on the CPU).
Replacing the first line above with

```julia
const T₀ = 20 # ᵒC
```

fixes the issue by indicating to the compiler that `T₀` will not change.

Note that the _literal_ `2π / 86400` is not an issue -- it's only the
_variable_ `T₀` that must be declared `const`.

### Complex diagnostics using `ComputedField`s may not work on GPUs

`ComputedField`s are the most convenient way to calculate diagnostics for your simulation. They will
always work on CPUs, but when their complexity is high (in terms of number of abstract operations)
the compiler can't translate them into GPU code and they fail for GPU runs. (This limitation is discussed 
in [this Github issue](https://github.com/CliMA/Oceananigans.jl/issues/1241) and contributors are welcome.)
For example, in the example below, calculating `u²` works in both CPUs and GPUs, but calculating 
`ε` will not compile on GPUs when we call the command `compute!`:

```julia
using Oceananigans
grid = RegularRectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
model = IncompressibleModel(grid=grid, closure=IsotropicDiffusivity(ν=1e-6))
u, v, w = model.velocities
ν = model.closure.ν
u² = ComputedField(u^2)
ε = ComputedField(ν*(∂x(u)^2 + ∂x(v)^2 + ∂x(w)^2 + ∂y(u)^2 + ∂y(v)^2 + ∂y(w)^2 + ∂z(u)^2 + ∂z(v)^2 + ∂z(w)^2))
compute!(u²)
compute!(ε)
```

There are two approaches to 
bypass this issue. The first is to nest `ComputedField`s. For example,
we can make `ε` be successfully computed on GPUs by defining it as
```julia
ddx² = ComputedField(∂x(u)^2 + ∂x(v)^2 + ∂x(w)^2)
ddy² = ComputedField(∂y(u)^2 + ∂y(v)^2 + ∂y(w)^2)
ddz² = ComputedField(∂z(u)^2 + ∂z(v)^2 + ∂z(w)^2)
ε = ComputedField(ν*(ddx² + ddy² + ddz²))
compute!(ε)
```

This is a simple workaround that is especially suited for the development stage of a simulation.
However, when running this, the code will iterate over the whole domain 4 times to calculate `ε`
(one for each computed field defined), which is not very efficient and may slow down your simulation
if this diagnostic is being calculated very often.

A different way to calculate `ε` is by using `KernelFunctionOperations`s, where the
user manually specifies the computing kernel function to the compiler. The advantage of this method is that
it's more efficient (the code will only iterate once over the domain in order to calculate `ε`),
but the disadvantage is that this method requires some knowledge of Oceananigans operations
and how they should be performed on a C-grid. For example calculating `ε` with this approach would
look like this:

```julia
using Oceananigans.Operators
using Oceananigans.AbstractOperations: KernelFunctionOperation

@inline fψ_plus_gφ²(i, j, k, grid, f, ψ, g, φ) = @inbounds (f(i, j, k, grid, ψ) + g(i, j, k, grid, φ))^2
function isotropic_viscous_dissipation_rate_ccc(i, j, k, grid, u, v, w, ν)
    Σˣˣ² = ∂xᶜᵃᵃ(i, j, k, grid, u)^2
    Σʸʸ² = ∂yᵃᶜᵃ(i, j, k, grid, v)^2
    Σᶻᶻ² = ∂zᵃᵃᶜ(i, j, k, grid, w)^2

    Σˣʸ² = ℑxyᶜᶜᵃ(i, j, k, grid, fψ_plus_gφ², ∂yᵃᶠᵃ, u, ∂xᶠᵃᵃ, v) / 4
    Σˣᶻ² = ℑxzᶜᵃᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᵃᵃᶠ, u, ∂xᶠᵃᵃ, w) / 4
    Σʸᶻ² = ℑyzᵃᶜᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᵃᵃᶠ, v, ∂yᵃᶠᵃ, w) / 4

    return ν * 2 * (Σˣˣ² + Σʸʸ² + Σᶻᶻ² + 2 * (Σˣʸ² + Σˣᶻ² + Σʸᶻ²))
end
ε = ComputedField(KernelFunctionOperation{Center, Center, Center}(isotropic_viscous_dissipation_rate_ccc, grid;
                         computed_dependencies=(u, v, w, ν)))
compute!(ε)
```


It may be useful to know that there are some kernels already defined for commonly-used diagnostics
in packages that are companions to Oceananigans. For example
[Oceanostics.jl](https://github.com/tomchor/Oceanostics.jl/blob/3b8f67338656557877ef8ef5ebe3af9e7b2974e2/src/TurbulentKineticEnergyTerms.jl#L35-L57)
and
[LESbrary.jl](https://github.com/CliMA/LESbrary.jl/blob/master/src/TurbulenceStatistics/shear_production.jl).
Users should first look there before writing any kernel by hand and are always welcome to [start an
issue on Github](https://github.com/CliMA/Oceananigans.jl/issues/new) if they need help to write a
different kernel. As an illustration, the calculation of `ε` using Oceanostics.jl (after installing the package)
which works on both CPUs and GPUs is simply

```julia
using Oceanostics: IsotropicPseudoViscousDissipationRate
ε = IsotropicViscousDissipationRate(model, u, v, w, ν)
compute!(ε)
```


### Try to decrease the memory-use of your runs

GPU runs are generally memory-limited. As an example, a state-of-the-art Tesla V100 GPU has 32GB of
memory, which is enough to fit, on average, a simulation with about 100 million points --- a bit
smaller than a 512-cubed simulation. (The precise number depends on many other things, such as the
number of tracers simulated, as well as the diagnostics that are calculated.) This means that it is
especially important to be mindful of the size of your runs when running Oceananigans on GPUs and it
is generally good practice to decrease the memory required for your runs. Below are some useful tips
to achieve this

- Use the [`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface) command
  line utility to monitor the memory usage of the GPU. It should tell you how much memory there is
  on your GPU and how much of it you're using and you can run it from Julia with the command ``run(`nvidia-smi`)``.
- Try to use higher-order advection schemes. In general when you use a higher-order scheme you need
  fewer grid points to achieve the same accuracy that you would with a lower-order one. Oceananigans
  provides two high-order advection schemes: 5th-order WENO method (WENO5) and 3rd-order upwind.
- Manually define scratch space to be reused in diagnostics. By default, every time a user-defined
  diagnostic is calculated the compiler reserves a new chunk of memory for that calculation, usually
  called scratch space. In general, the more diagnostics, the more scratch space needed and the bigger
  the memory requirements. However, if you explicitly create a scratch space and pass that same
  scratch space for as many diagnostics as you can, you minimize the memory requirements of your
  calculations by reusing the same chunk of memory. As an example, you can see scratch space being
  created
  [here](https://github.com/CliMA/LESbrary.jl/blob/cf31b0ec20219d5ad698af334811d448c27213b0/examples/three_layer_ constant_fluxes.jl#L380-L383)
  and then being used in calculations
  [here](https://github.com/CliMA/LESbrary.jl/blob/cf31b0ec20219d5ad698af334811d448c27213b0/src/TurbulenceStatistics/first_through_third_order.jl#L109-L112).



### Arrays in GPUs are usually different from arrays in CPUs

Oceananigans.jl uses [`CUDA.CuArray`](https://cuda.juliagpu.org/stable/usage/array/) to store 
data for GPU computations. One limitation of `CuArray`s compared to the `Array`s used for 
CPU computations is that `CuArray` elements in general cannot be accessed outside kernels
launched through CUDA.jl or KernelAbstractions.jl. (You can learn more about GPU kernels 
[here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels) and 
[here](https://cuda.juliagpu.org/stable/usage/overview/#Kernel-programming-with-@cuda).)
Doing so requires individual elements to be copied from or to the GPU for processing,
which is very slow and can result in huge slowdowns. For this reason, Oceananigans.jl disables CUDA
scalar indexing by default. See the [scalar indexing](https://juliagpu.github.io/CUDA.jl/dev/usage/workflow/#UsageWorkflowScalar)
section of the CUDA.jl documentation for more information on scalar indexing.


For example if can be difficult to just view a `CuArray` since Julia needs to access 
its elements to do that. Consider the example below:

```julia
julia> using Oceananigans; using Adapt

julia> grid = RegularRectilinearGrid(size=(1,1,1), extent=(1,1,1))
RegularRectilinearGrid{Float64, Periodic, Periodic, Bounded}
                   domain: x ∈ [0.0, 1.0], y ∈ [0.0, 1.0], z ∈ [-1.0, 0.0]
                 topology: (Periodic, Periodic, Bounded)
  resolution (Nx, Ny, Nz): (1, 1, 1)
   halo size (Hx, Hy, Hz): (1, 1, 1)
grid spacing (Δx, Δy, Δz): (1.0, 1.0, 1.0)

julia> model = IncompressibleModel(grid=grid, architecture=GPU())
IncompressibleModel{GPU, Float64}(time = 0 seconds, iteration = 0) 
├── grid: RegularRectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=1, Ny=1, Nz=1)
├── tracers: (:T, :S)
├── closure: IsotropicDiffusivity{Float64,NamedTuple{(:T, :S),Tuple{Float64,Float64}}}
├── buoyancy: SeawaterBuoyancy{Float64,LinearEquationOfState{Float64},Nothing,Nothing}
└── coriolis: Nothing

julia> typeof(model.velocities.u.data)
OffsetArrays.OffsetArray{Float64,3,CUDA.CuArray{Float64,3}}

julia> adapt(Array, model.velocities.u.data)
3×3×3 OffsetArray(::Array{Float64,3}, 0:2, 0:2, 0:2) with eltype Float64 with indices 0:2×0:2×0:2:
[:, :, 0] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 1] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 2] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
```

Notice that in order to view the `CuArray` that stores values for `u` we needed to transform
it into a regular `Array` first using `Adapt.adapt`. If we naively try to view the `CuArray`
without that step we get an error:

```julia
julia> model.velocities.u.data
3×3×3 OffsetArray(::CUDA.CuArray{Float64,3}, 0:2, 0:2, 0:2) with eltype Float64 with indices 0:2×0:2×0:2:
[:, :, 0] =
Error showing value of type OffsetArrays.OffsetArray{Float64,3,CUDA.CuArray{Float64,3}}:
ERROR: scalar getindex is disallowed
```

Here `CUDA.jl` throws an error because scalar `getindex` is not `allowed`. Another way around 
this limitation is to allow scalar operations on `CuArray`s. We can temporarily
do that with the `CUDA.@allowscalar` macro or by calling `CUDA.allowscalar(true)`.


```julia
julia> using CUDA; CUDA.allowscalar(true)

julia> model.velocities.u.data
3×3×3 OffsetArray(::CuArray{Float64,3}, 0:2, 0:2, 0:2) with eltype Float64 with indices 0:2×0:2×0:2:
[:, :, 0] =
┌ Warning: Performing scalar operations on GPU arrays: This is very slow, consider disallowing these operations with `allowscalar(false)`
└ @ GPUArrays ~/.julia/packages/GPUArrays/WV76E/src/host/indexing.jl:43
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 1] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 2] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
```

Notice the warning we get when we do this. Scalar operations on GPUs can be very slow, so it is
advised to only use this last method when using the REPL or prototyping --- never in
production-ready scripts.


You might also need to keep these differences in mind when using arrays
to define initial conditions, boundary conditions or
forcing functions on a GPU. To learn more about working with `CuArray`s, see the
[array programming](https://juliagpu.github.io/CUDA.jl/dev/usage/array/) section
of the CUDA.jl documentation.

