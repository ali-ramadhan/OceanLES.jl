#####
##### The turbulence closure proposed by Smagorinsky and Lilly.
##### We also call this 'Constant Smagorinsky'.
#####

struct SmagorinskyLilly{TD, FT, P, K} <: AbstractEddyViscosityClosure{TD}
     C :: FT
    Cb :: FT
    Pr :: P
     ν :: FT
     κ :: K

    function SmagorinskyLilly{TD, FT}(C, Cb, Pr, ν, κ) where {TD, FT}
        Pr = convert_diffusivity(FT, Pr)
         κ = convert_diffusivity(FT, κ)
        return new{TD, FT, typeof(Pr), typeof(κ)}(C, Cb, Pr, ν, κ)
    end
end

"""
    SmagorinskyLilly([FT=Float64;] C=0.23, Pr=1, ν=0, κ=0,
                                   time_discretization=ExplicitTimeDiscretization())

Return a `SmagorinskyLilly` type associated with the turbulence closure proposed by
Lilly (1962) and Smagorinsky (1958, 1963), which has an eddy viscosity of the form

    `νₑ = (C * Δᶠ)² * √(2Σ²) * √(1 - Cb * N² / Σ²) + ν`,

and an eddy diffusivity of the form

    `κₑ = (νₑ - ν) / Pr + κ`

where `Δᶠ` is the filter width, `Σ² = ΣᵢⱼΣᵢⱼ` is the double dot product of
the strain tensor `Σᵢⱼ`, `Pr` is the turbulent Prandtl number, and `N²` is
the total buoyancy gradient, and `Cb` is a constant the multiplies the Richardson number
modification to the eddy viscosity.

Keyword arguments
=================
    - `C`  : Model constant
    - `Cb` : Buoyancy term multipler (`Cb = 0` turns it off, `Cb ≠ 0` turns it on.
             Typically `Cb=1/Pr`.)
    - `Pr` : Turbulent Prandtl numbers for each tracer. Either a constant applied to every
             tracer, or a `NamedTuple` with fields for each tracer individually.
    - `ν`  : Constant background viscosity for momentum
    - `κ`  : Constant background diffusivity for tracer. Can either be a single number
             applied to all tracers, or `NamedTuple` of diffusivities corresponding to each
             tracer.
    - `time_discretization` : Either `ExplicitTimeDiscretization()` or `VerticallyImplicitTimeDiscretization()`, 
                              which integrates the terms involving only z-derivatives in the
                              viscous and diffusive fluxes with an implicit time discretization.

References
==========
Smagorinsky, J. "On the numerical integration of the primitive equations of motion for
    baroclinic flow in a closed region." Monthly Weather Review (1958)

Lilly, D. K. "On the numerical simulation of buoyant convection." Tellus (1962)

Smagorinsky, J. "General circulation experiments with the primitive equations: I.
    The basic experiment." Monthly weather review (1963)
"""
SmagorinskyLilly(FT=Float64; C=0.23, Cb=1.0, Pr=1.0, ν=0, κ=0,
                             time_discretization::TD=ExplicitTimeDiscretization()) where TD =
    SmagorinskyLilly{TD, FT}(C, Cb, Pr, ν, κ)

function with_tracers(tracers, closure::SmagorinskyLilly{TD, FT}) where {TD, FT}
    Pr = tracer_diffusivities(tracers, closure.Pr)
     κ = tracer_diffusivities(tracers, closure.κ)
    return SmagorinskyLilly{TD, FT}(closure.C, closure.Cb, Pr, closure.ν, κ)
end

"""
    stability(N², Σ², Cb)

Return the stability function

    ``\$ \\sqrt(1 - Cb N^2 / Σ^2 ) \$``

when ``N^2 > 0``, and 1 otherwise.
"""
@inline stability(N²::FT, Σ²::FT, Cb::FT) where FT =
    ifelse(Σ²==0, zero(FT), sqrt(one(FT) - stability_factor(N², Σ², Cb)))

@inline stability_factor(N²::FT, Σ²::FT, Cb::FT) where FT = min(one(FT), Cb * N² / Σ²)

"""
    νₑ_deardorff(ς, C, Δᶠ, Σ²)

Return the eddy viscosity for constant Smagorinsky
given the stability `ς`, model constant `C`,
filter width `Δᶠ`, and strain tensor dot product `Σ²`.
"""
@inline νₑ_deardorff(ς, C, Δᶠ, Σ²) = ς * (C*Δᶠ)^2 * sqrt(2Σ²)

@inline function νᶜᶜᶜ(i, j, k, grid::AbstractGrid{FT}, clo::SmagorinskyLilly, buoyancy, U, C) where FT
    Σ² = ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, U.u, U.v, U.w)
    N² = max(zero(FT), ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, C))
    Δᶠ = Δᶠ_ccc(i, j, k, grid, clo)
     ς = stability(N², Σ², clo.Cb) # Use unity Prandtl number.

    return νₑ_deardorff(ς, clo.C, Δᶠ, Σ²) + clo.ν
end

function calculate_diffusivities!(K, arch, grid, closure::SmagorinskyLilly, buoyancy, U, C)

    event = launch!(arch, grid, :xyz, calculate_nonlinear_viscosity!, K.νₑ, grid, closure, buoyancy, U, C,
                    dependencies=Event(device(arch)))

    wait(device(arch), event)

    return nothing
end

#####
##### Double dot product of strain on cell edges (currently unused)
#####

"Return the filter width for Constant Smagorinsky on a regular rectilinear grid."
@inline Δᶠ(i, j, k, grid, ::SmagorinskyLilly) = geo_mean_Δᶠ(i, j, k, grid)

# Temporarily set filter widths to cell-size (rather than distance between cell centers, etc.)
const Δᶠ_ccc = Δᶠ
const Δᶠ_ccf = Δᶠ
const Δᶠ_ffc = Δᶠ
const Δᶠ_fcf = Δᶠ
const Δᶠ_cff = Δᶠ

# tr_Σ² : ccc
#   Σ₁₂ : ffc
#   Σ₁₃ : fcf
#   Σ₂₃ : cff

"Return the double dot product of strain at `ccc`."
@inline function ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)
    return (
                    tr_Σ²(i, j, k, grid, u, v, w)
            + 2 * ℑxyᶜᶜᵃ(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 * ℑxzᶜᵃᶜ(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 * ℑyzᵃᶜᶜ(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `ffc`."
@inline function ΣᵢⱼΣᵢⱼᶠᶠᶜ(i, j, k, grid, u, v, w)
    return (
                  ℑxyᶠᶠᵃ(i, j, k, grid, tr_Σ², u, v, w)
            + 2 *    Σ₁₂²(i, j, k, grid, u, v, w)
            + 2 * ℑyzᵃᶠᶜ(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 * ℑxzᶠᵃᶜ(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `fcf`."
@inline function ΣᵢⱼΣᵢⱼᶠᶜᶠ(i, j, k, grid, u, v, w)
    return (
                  ℑxzᶠᵃᶠ(i, j, k, grid, tr_Σ², u, v, w)
            + 2 * ℑyzᵃᶜᶠ(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 *    Σ₁₃²(i, j, k, grid, u, v, w)
            + 2 * ℑxyᶠᶜᵃ(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `cff`."
@inline function ΣᵢⱼΣᵢⱼᶜᶠᶠ(i, j, k, grid, u, v, w)
    return (
                  ℑyzᵃᶠᶠ(i, j, k, grid, tr_Σ², u, v, w)
            + 2 * ℑxzᶜᵃᶠ(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 * ℑxyᶜᶠᵃ(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 *    Σ₂₃²(i, j, k, grid, u, v, w)
            )
end

@inline function ΣᵢⱼΣᵢⱼᶜᶜᶠ(i, j, k, grid, u, v, w)
    return (
                    ℑzᵃᵃᶠ(i, j, k, grid, tr_Σ², u, v, w)
            + 2 * ℑxyzᶜᶜᶠ(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 *   ℑxᶜᵃᵃ(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 *   ℑyᵃᶜᵃ(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

Base.show(io::IO, closure::SmagorinskyLilly) =
    print(io, "SmagorinskyLilly: C=$(closure.C), Cb=$(closure.Cb), Pr=$(closure.Pr), ν=$(closure.ν), κ=$(closure.κ)")

#####
##### For closures that only require an eddy viscosity νₑ field.
#####

function DiffusivityFields(arch, grid, tracer_names, bcs, closure::SmagorinskyLilly)

    default_eddy_viscosity_bcs = (; νₑ = FieldBoundaryConditions(grid, (Center, Center, Center)))
    bcs = merge(default_eddy_viscosity_bcs, bcs)
    νₑ = CenterField(arch, grid, bcs.νₑ)

    # Use AbstractOperations to write eddy diffusivities in terms of
    # eddy viscosity
    κₑ_ops = []

    for i = 1:length(tracer_names)
        Pr = closure.Pr[i]
        κ = closure.κ[i]
        ν = closure.ν
        κₑ_op = (νₑ - ν) / Pr + κ
        push!(κₑ_ops, κₑ_op)
    end

    κₑ = NamedTuple{tracer_names}(Tuple(κₑ_ops))

    return (; νₑ, κₑ)
end

