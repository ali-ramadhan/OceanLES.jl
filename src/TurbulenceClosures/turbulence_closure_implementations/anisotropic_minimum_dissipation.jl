using Oceananigans.Operators

"""
    AnisotropicMinimumDissipation{FT} <: AbstractTurbulenceClosure

Parameters for the "anisotropic minimum dissipation" turbulence closure for large eddy simulation
proposed originally by [Rozema15](@cite) and [Abkar16](@cite), and then modified
by [Verstappen18](@cite), and finally described and validated for by [Vreugdenhil18](@cite).
"""
struct AnisotropicMinimumDissipation{TD, FT, PK, PN, K, PB} <: AbstractEddyViscosityClosure{TD}
    Cν :: PN
    Cκ :: PK
    Cb :: PB
     ν :: FT
     κ :: K

    function AnisotropicMinimumDissipation{TD, FT}(Cν::PN, Cκ::PK, Cb::PB, ν, κ) where {TD, FT, PN, PK, PB}
        κ = convert_diffusivity(FT, κ)
        K = typeof(κ)
        return new{TD, FT, PK, PN, K, PB}(Cν, Cκ, Cb, ν, κ)
    end
end

const AMD = AnisotropicMinimumDissipation

Base.show(io::IO, closure::AMD{TD, FT}) where {TD, FT} =
    print(io, "AnisotropicMinimumDissipation{$FT} turbulence closure with:\n",
              "           Poincaré constant for momentum eddy viscosity Cν: ", closure.Cν, '\n',
              "    Poincaré constant for tracer(s) eddy diffusivit(ies) Cκ: ", closure.Cκ, '\n',
              "                        Buoyancy modification multiplier Cb: ", closure.Cb, '\n',
              "                Background diffusivit(ies) for tracer(s), κ: ", closure.κ, '\n',
              "             Background kinematic viscosity for momentum, ν: ", closure.ν)

"""
    AnisotropicMinimumDissipation([FT=Float64;] C=1/12, Cν=nothing, Cκ=nothing,
                                                Cb=nothing, ν=0, κ=0,
                                                time_discretization=ExplicitTimeDiscretization())
                                       
Returns parameters of type `FT` for the `AnisotropicMinimumDissipation`
turbulence closure.

Keyword arguments
=================
    - `C` : Poincaré constant for both eddy viscosity and eddy diffusivities. `C` is overridden
            for eddy viscosity or eddy diffusivity if `Cν` or `Cκ` are set, respecitvely.

    - `Cν` : Poincaré constant for momentum eddy viscosity.

    - `Cκ` : Poincaré constant for tracer eddy diffusivities. If one number or function, the same
             number or function is applied to all tracers. If a `NamedTuple`, it must possess
             a field specifying the Poncaré constant for every tracer.

    - `Cb` : Buoyancy modification multiplier (`Cb = nothing` turns it off, `Cb = 1` was used by [Abkar16](@cite)).
             *Note*: that we _do not_ subtract the horizontally-average component before computing this
             buoyancy modification term. This implementation differs from [Abkar16](@cite)'s proposal
             and the impact of this approximation has not been tested or validated.

    - `ν` : Constant background viscosity for momentum.

    - `κ` : Constant background diffusivity for tracer. If a single number, the same background
            diffusivity is applied to all tracers. If a `NamedTuple`, it must possess a field
            specifying a background diffusivity for every tracer.

    - `time_discretization` : Either `ExplicitTimeDiscretization()` or `VerticallyImplicitTimeDiscretization()`, 
                              which integrates the terms involving only z-derivatives in the
                              viscous and diffusive fluxes with an implicit time discretization.

By default: `C = Cν = Cκ` = 1/12, which is appropriate for a finite-volume method employing a
second-order advection scheme, `Cb = nothing`, which terms off the buoyancy modification term.

`Cν` or `Cκ` may be constant numbers, or functions of `x, y, z`.

Example
=======

```julia
julia> pretty_diffusive_closure = AnisotropicMinimumDissipation(C=1/2)
AnisotropicMinimumDissipation{Float64} turbulence closure with:
           Poincaré constant for momentum eddy viscosity Cν: 0.5
    Poincaré constant for tracer(s) eddy diffusivit(ies) Cκ: 0.5
                        Buoyancy modification multiplier Cb: nothing
                Background diffusivit(ies) for tracer(s), κ: 0.0
             Background kinematic viscosity for momentum, ν: 0.0

julia> const Δz = 0.5; # grid resolution at surface

julia> surface_enhanced_tracer_C(x, y, z) = 1/12 * (1 + exp((z + Δz/2) / 8Δz))
surface_enhanced_tracer_C (generic function with 1 method)

julia> fancy_closure = AnisotropicMinimumDissipation(Cκ=surface_enhanced_tracer_C)
AnisotropicMinimumDissipation{Float64} turbulence closure with:
           Poincaré constant for momentum eddy viscosity Cν: 0.08333333333333333
    Poincaré constant for tracer(s) eddy diffusivit(ies) Cκ: surface_enhanced_tracer_C
                        Buoyancy modification multiplier Cb: nothing
                Background diffusivit(ies) for tracer(s), κ: 0.0
             Background kinematic viscosity for momentum, ν: 0.0

julia> tracer_specific_closure = AnisotropicMinimumDissipation(Cκ=(c₁=1/12, c₂=1/6))
AnisotropicMinimumDissipation{Float64} turbulence closure with:
           Poincaré constant for momentum eddy viscosity Cν: 0.08333333333333333
    Poincaré constant for tracer(s) eddy diffusivit(ies) Cκ: (c₁ = 0.08333333333333333, c₂ = 0.16666666666666666)
                        Buoyancy modification multiplier Cb: nothing
                Background diffusivit(ies) for tracer(s), κ: 0.0
             Background kinematic viscosity for momentum, ν: 0.0
```

References
==========
Vreugdenhil C., and Taylor J. (2018), "Large-eddy simulations of stratified plane Couette
    flow using the anisotropic minimum-dissipation model", Physics of Fluids 30, 085104.

Verstappen, R. (2018), "How much eddy dissipation is needed to counterbalance the nonlinear
    production of small, unresolved scales in a large-eddy simulation of turbulence?",
    Computers & Fluids 176, pp. 276-284.
"""
function AnisotropicMinimumDissipation(FT = Float64;
                                       C = 1/12,
                                       Cν = nothing,
                                       Cκ = nothing,
                                       Cb = nothing,
                                       ν = 0,
                                       κ = 0,
                                       time_discretization::TD = ExplicitTimeDiscretization()) where TD
    Cν = Cν === nothing ? C : Cν
    Cκ = Cκ === nothing ? C : Cκ
    
    !isnothing(Cb) && @warn "AnisotropicMinimumDissipation with buoyancy modification is unvalidated."

    return AnisotropicMinimumDissipation{TD, FT}(Cν, Cκ, Cb, ν, κ)
end

function with_tracers(tracers, closure::AnisotropicMinimumDissipation{TD, FT}) where {TD, FT}
    κ = tracer_diffusivities(tracers, closure.κ)
    Cκ = tracer_diffusivities(tracers, closure.Cκ)
    return AnisotropicMinimumDissipation{TD, FT}(closure.Cν, Cκ, closure.Cb, closure.ν, κ)
end

#####
##### Kernel functions
#####

# Dispatch on the type of the user-provided AMD model constant.
# Only numbers, arrays, and functions supported now.
@inline Cᴾᵒⁱⁿ(i, j, k, grid, C::Number) = C
@inline Cᴾᵒⁱⁿ(i, j, k, grid, C::AbstractArray) = @inbounds C[i, j, k]
@inline Cᴾᵒⁱⁿ(i, j, k, grid, C::Function) = C(xnode(Center(), i, grid), ynode(Center(), j, grid), znode(Center(), k, grid))

@inline function νᶜᶜᶜ(i, j, k, grid::AbstractGrid{FT}, closure::AMD, buoyancy, U, C) where FT
    ijk = (i, j, k, grid)
    q = norm_tr_∇uᶜᶜᶜ(ijk..., U.u, U.v, U.w)
    Cb = closure.Cb

    if q == 0 # SGS viscosity is zero when strain is 0
        νˢᵍˢ = zero(FT)
    else
        r = norm_uᵢₐ_uⱼₐ_Σᵢⱼᶜᶜᶜ(ijk..., closure, U.u, U.v, U.w)

        # So-called buoyancy modification term:
        Cb_ζ = Cb_norm_wᵢ_bᵢᶜᶜᶜ(ijk..., Cb, closure, buoyancy, U.w, C) / Δᶠzᶜᶜᶜ(ijk...)

        δ² = 3 / (1 / Δᶠxᶜᶜᶜ(ijk...)^2 + 1 / Δᶠyᶜᶜᶜ(ijk...)^2 + 1 / Δᶠzᶜᶜᶜ(ijk...)^2)

        νˢᵍˢ = - Cᴾᵒⁱⁿ(i, j, k, grid, closure.Cν) * δ² * (r - Cb_ζ) / q
    end

    return max(zero(FT), νˢᵍˢ) + closure.ν
end

@inline function κᶜᶜᶜ(i, j, k, grid::AbstractGrid{FT}, closure::AMD, c, ::Val{tracer_index},
                       U) where {FT, tracer_index}

    ijk = (i, j, k, grid)

    @inbounds κ = closure.κ[tracer_index]
    @inbounds Cκ = closure.Cκ[tracer_index]

    σ =  norm_θᵢ²ᶜᶜᶜ(i, j, k, grid, c)

    if σ == 0 # denominator is zero: short-circuit computations and set subfilter diffusivity to zero.
        κˢᵍˢ = zero(FT)
    else
        ϑ =  norm_uᵢⱼ_cⱼ_cᵢᶜᶜᶜ(ijk..., closure, U.u, U.v, U.w, c)
        δ² = 3 / (1 / Δᶠxᶜᶜᶜ(ijk...)^2 + 1 / Δᶠyᶜᶜᶜ(ijk...)^2 + 1 / Δᶠzᶜᶜᶜ(ijk...)^2)
        κˢᵍˢ = - Cᴾᵒⁱⁿ(i, j, k, grid, Cκ) * δ² * ϑ / σ
    end

    return max(zero(FT), κˢᵍˢ) + κ
end

function calculate_diffusivities!(K, arch, grid, closure::AnisotropicMinimumDissipation, buoyancy, U, C)
    workgroup, worksize = work_layout(grid, :xyz)

    barrier = Event(device(arch))

    viscosity_kernel! = calculate_viscosity!(device(arch), workgroup, worksize)
    diffusivity_kernel! = calculate_tracer_diffusivity!(device(arch), workgroup, worksize)

    viscosity_event = viscosity_kernel!(K.νₑ, grid, closure, buoyancy, U, C, dependencies=barrier)

    events = [viscosity_event]

    for (tracer_index, κₑ) in enumerate(K.κₑ)
        @inbounds c = C[tracer_index]
        event = diffusivity_kernel!(κₑ, grid, closure, c, Val(tracer_index), U, dependencies=barrier)
        push!(events, event)
    end

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

@kernel function calculate_viscosity!(νₑ, grid, closure::AnisotropicMinimumDissipation, buoyancy, U, C)
    i, j, k = @index(Global, NTuple)
    @inbounds νₑ[i, j, k] = νᶜᶜᶜ(i, j, k, grid, closure, buoyancy, U, C)
end

@kernel function calculate_tracer_diffusivity!(κₑ, grid, closure, c, tracer_index, U)
    i, j, k = @index(Global, NTuple)
    @inbounds κₑ[i, j, k] = κᶜᶜᶜ(i, j, k, grid, closure, c, tracer_index, U)
end

#####
##### Filter width at various locations
#####

# Recall that filter widths are 2x the grid spacing in AMD
@inline Δᶠxᶜᶜᶜ(i, j, k, grid) = 2 * Δxᶜᶜᵃ(i, j, k, grid)
@inline Δᶠyᶜᶜᶜ(i, j, k, grid) = 2 * Δyᶜᶜᵃ(i, j, k, grid)
@inline Δᶠzᶜᶜᶜ(i, j, k, grid) = 2 * Δzᵃᵃᶜ(i, j, k, grid)

for loc in (:ccf, :fcc, :cfc, :ffc, :cff, :fcf), ξ in (:x, :y, :z)
    Δ_loc = Symbol(:Δᶠ, ξ, :_, loc)
    Δᶜᶜᶜ = Symbol(:Δᶠ, ξ, :ᶜᶜᶜ)
    @eval begin
        const $Δ_loc = $Δᶜᶜᶜ
    end
end

#####
##### The *** 30 terms *** of AMD
#####

@inline function norm_uᵢₐ_uⱼₐ_Σᵢⱼᶜᶜᶜ(i, j, k, grid, closure, u, v, w)
    ijk = (i, j, k, grid)
    uvw = (u, v, w)
    ijkuvw = (i, j, k, grid, u, v, w)

    uᵢ₁_uⱼ₁_Σ₁ⱼ = (
         norm_Σ₁₁(ijkuvw...) * norm_∂x_u(ijk..., u)^2
      +  norm_Σ₂₂(ijkuvw...) * ℑxyᶜᶜᵃ(ijk..., norm_∂x_v², uvw...)
      +  norm_Σ₃₃(ijkuvw...) * ℑxzᶜᵃᶜ(ijk..., norm_∂x_w², uvw...)

      +  2 * norm_∂x_u(ijkuvw...) * ℑxyᶜᶜᵃ(ijk..., norm_∂x_v_Σ₁₂, uvw...)
      +  2 * norm_∂x_u(ijkuvw...) * ℑxzᶜᵃᶜ(ijk..., norm_∂x_w_Σ₁₃, uvw...)
      +  2 * ℑxyᶜᶜᵃ(ijk..., norm_∂x_v, uvw...) * ℑxzᶜᵃᶜ(ijk..., norm_∂x_w, uvw...)
           * ℑyzᵃᶜᶜ(ijk..., norm_Σ₂₃, uvw...)
    )

    uᵢ₂_uⱼ₂_Σ₂ⱼ = (
      + norm_Σ₁₁(ijkuvw...) * ℑxyᶜᶜᵃ(ijk..., norm_∂y_u², uvw...)
      + norm_Σ₂₂(ijkuvw...) * norm_∂y_v(ijk..., v)^2
      + norm_Σ₃₃(ijkuvw...) * ℑyzᵃᶜᶜ(ijk..., norm_∂y_w², uvw...)

      +  2 * norm_∂y_v(ijkuvw...) * ℑxyᶜᶜᵃ(ijk..., norm_∂y_u_Σ₁₂, uvw...)
      +  2 * ℑxyᶜᶜᵃ(ijk..., norm_∂y_u, uvw...) * ℑyzᵃᶜᶜ(ijk..., norm_∂y_w, uvw...)
           * ℑxzᶜᵃᶜ(ijk..., norm_Σ₁₃, uvw...)
      +  2 * norm_∂y_v(ijkuvw...) * ℑyzᵃᶜᶜ(ijk..., norm_∂y_w_Σ₂₃, uvw...)
    )

    uᵢ₃_uⱼ₃_Σ₃ⱼ = (
      + norm_Σ₁₁(ijkuvw...) * ℑxzᶜᵃᶜ(ijk..., norm_∂z_u², uvw...)
      + norm_Σ₂₂(ijkuvw...) * ℑyzᵃᶜᶜ(ijk..., norm_∂z_v², uvw...)
      + norm_Σ₃₃(ijkuvw...) * norm_∂z_w(ijk..., w)^2

      +  2 * ℑxzᶜᵃᶜ(ijk..., norm_∂z_u, uvw...) * ℑyzᵃᶜᶜ(ijk..., norm_∂z_v, uvw...)
           * ℑxyᶜᶜᵃ(ijk..., norm_Σ₁₂, uvw...)
      +  2 * norm_∂z_w(ijkuvw...) * ℑxzᶜᵃᶜ(ijk..., norm_∂z_u_Σ₁₃, uvw...)
      +  2 * norm_∂z_w(ijkuvw...) * ℑyzᵃᶜᶜ(ijk..., norm_∂z_v_Σ₂₃, uvw...)
    )

    return uᵢ₁_uⱼ₁_Σ₁ⱼ + uᵢ₂_uⱼ₂_Σ₂ⱼ + uᵢ₃_uⱼ₃_Σ₃ⱼ
end

#####
##### trace(∇u) = uᵢⱼ uᵢⱼ
#####

@inline function norm_tr_∇uᶜᶜᶜ(i, j, k, grid, uvw...)
    ijk = (i, j, k, grid)

    return (
        # ccc
        norm_∂x_u²(ijk..., uvw...)
      + norm_∂y_v²(ijk..., uvw...)
      + norm_∂z_w²(ijk..., uvw...)

        # ffc
      + ℑxyᶜᶜᵃ(ijk..., norm_∂x_v², uvw...)
      + ℑxyᶜᶜᵃ(ijk..., norm_∂y_u², uvw...)

        # fcf
      + ℑxzᶜᵃᶜ(ijk..., norm_∂x_w², uvw...)
      + ℑxzᶜᵃᶜ(ijk..., norm_∂z_u², uvw...)

        # cff
      + ℑyzᵃᶜᶜ(ijk..., norm_∂y_w², uvw...)
      + ℑyzᵃᶜᶜ(ijk..., norm_∂z_v², uvw...)
    )
end

@inline Cb_norm_wᵢ_bᵢᶜᶜᶜ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, args...) where FT = zero(FT)

@inline function Cb_norm_wᵢ_bᵢᶜᶜᶜ(i, j, k, grid, Cb, closure, buoyancy, w, C)
    ijk = (i, j, k, grid)

    wx_bx = (ℑxzᶜᵃᶜ(ijk..., norm_∂x_w, w)
             * Δᶠxᶜᶜᶜ(ijk...) * ℑxᶜᵃᵃ(ijk..., ∂xᶠᵃᵃ, buoyancy_perturbation, buoyancy.model, C))

    wy_by = (ℑyzᵃᶜᶜ(ijk..., norm_∂y_w, w)
             * Δᶠyᶜᶜᶜ(ijk...) * ℑyᵃᶜᵃ(ijk..., ∂yᵃᶠᵃ, buoyancy_perturbation, buoyancy.model, C))

    wz_bz = (norm_∂z_w(ijk..., w)
             * Δᶠzᶜᶜᶜ(ijk...) * ℑzᵃᵃᶜ(ijk..., ∂zᵃᵃᶠ, buoyancy_perturbation, buoyancy.model, C))

    return Cb * (wx_bx + wy_by + wz_bz)
end

@inline function norm_uᵢⱼ_cⱼ_cᵢᶜᶜᶜ(i, j, k, grid, closure, u, v, w, c)
    ijk = (i, j, k, grid)

    cx_ux = (
                  norm_∂x_u(ijk..., u) * ℑxᶜᵃᵃ(ijk..., norm_∂x_c², c)
        + ℑxyᶜᶜᵃ(ijk..., norm_∂x_v, v) * ℑxᶜᵃᵃ(ijk..., norm_∂x_c, c) * ℑyᵃᶜᵃ(ijk..., norm_∂y_c, c)
        + ℑxzᶜᵃᶜ(ijk..., norm_∂x_w, w) * ℑxᶜᵃᵃ(ijk..., norm_∂x_c, c) * ℑzᵃᵃᶜ(ijk..., norm_∂z_c, c)
    )

    cy_uy = (
          ℑxyᶜᶜᵃ(ijk..., norm_∂y_u, u) * ℑyᵃᶜᵃ(ijk..., norm_∂y_c, c) * ℑxᶜᵃᵃ(ijk..., norm_∂x_c, c)
        +         norm_∂y_v(ijk..., v) * ℑyᵃᶜᵃ(ijk..., norm_∂y_c², c)
        + ℑxzᶜᵃᶜ(ijk..., norm_∂y_w, w) * ℑyᵃᶜᵃ(ijk..., norm_∂y_c, c) * ℑzᵃᵃᶜ(ijk..., norm_∂z_c, c)
    )

    cz_uz = (
          ℑxzᶜᵃᶜ(ijk..., norm_∂z_u, u) * ℑzᵃᵃᶜ(ijk..., norm_∂z_c, c) * ℑxᶜᵃᵃ(ijk..., norm_∂x_c, c)
        + ℑyzᵃᶜᶜ(ijk..., norm_∂z_v, v) * ℑzᵃᵃᶜ(ijk..., norm_∂z_c, c) * ℑyᵃᶜᵃ(ijk..., norm_∂y_c, c)
        +         norm_∂z_w(ijk..., w) * ℑzᵃᵃᶜ(ijk..., norm_∂z_c², c)
    )

    return cx_ux + cy_uy + cz_uz
end

@inline norm_θᵢ²ᶜᶜᶜ(i, j, k, grid, c) = ℑxᶜᵃᵃ(i, j, k, grid, norm_∂x_c², c) +
                                        ℑyᵃᶜᵃ(i, j, k, grid, norm_∂y_c², c) +
                                        ℑzᵃᵃᶜ(i, j, k, grid, norm_∂z_c², c)

#####
##### DiffusivityFields
#####

function DiffusivityFields(arch, grid, tracer_names, user_bcs, ::AMD)

    default_diffusivity_bcs = FieldBoundaryConditions(grid, (Center, Center, Center))
    default_κₑ_bcs = NamedTuple(c => default_diffusivity_bcs for c in tracer_names)
    κₑ_bcs = :κₑ ∈ keys(user_bcs) ? merge(default_κₑ_bcs, user_bcs.κₑ) : default_κₑ_bcs

    bcs = merge((; νₑ = default_diffusivity_bcs, κₑ = κₑ_bcs), user_bcs)

    νₑ = CenterField(arch, grid, bcs.νₑ)
    κₑ = NamedTuple(c => CenterField(arch, grid, bcs.κₑ[c]) for c in tracer_names)

    return (; νₑ, κₑ)
end

