using Oceananigans.Fields: AbstractField

#####
##### The turbulence closure proposed by Leith
#####

struct TwoDimensionalLeith{FT, CR, GM} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization}
         C :: FT
    C_Redi :: CR
      C_GM :: GM

    function TwoDimensionalLeith{FT}(C, C_Redi, C_GM) where FT
        C_Redi = convert_diffusivity(FT, C_Redi)
        C_GM = convert_diffusivity(FT, C_GM)
        return new{FT, typeof(C_Redi), typeof(C_GM)}(C, C_Redi, C_GM)
    end
end

"""
    TwoDimensionalLeith([FT=Float64;] C=0.3, C_Redi=1, C_GM=1)

Return a `TwoDimensionalLeith` type associated with the turbulence closure proposed by
Leith (1965) and Fox-Kemper & Menemenlis (2008) which has an eddy viscosity of the form

    `νₑ = (C * Δᶠ)³ * √(|∇h ζ|² + |∇h ∂z w|²)`

and an eddy diffusivity of the form...

where `Δᶠ` is the filter width, `ζ = ∂x v - ∂y u` is the vertical vorticity,
and `C` is a model constant.

Keyword arguments
=================
    - `C`      : Model constant
    - `C_Redi` : Coefficient for down-gradient tracer diffusivity for each tracer.
                 Either a constant applied to every tracer, or a `NamedTuple` with fields
                 for each tracer individually.
    - `C_GM`   : Coefficient for down-gradient tracer diffusivity for each tracer.
                 Either a constant applied to every tracer, or a `NamedTuple` with fields
                 for each tracer individually.

References
==========
Leith, C. E. (1968). "Diffusion Approximation for Two‐Dimensional Turbulence", The Physics of
    Fluids 11, 671. doi: 10.1063/1.1691968

Fox‐Kemper, B., & D. Menemenlis (2008), "Can large eddy simulation techniques improve mesoscale rich
    ocean models?", in Ocean Modeling in an Eddying Regime, Geophys. Monogr. Ser., vol. 177, pp. 319–337.
    doi: 10.1029/177GM19

Pearson, B. et al. (2017) , "Evaluation of scale-aware subgrid mesoscale eddy models in a global eddy
    rich model", Ocean Modelling 115, 42-58. doi: 10.1016/j.ocemod.2017.05.007
"""
TwoDimensionalLeith(FT=Float64; C=0.3, C_Redi=1, C_GM=1) = TwoDimensionalLeith{FT}(C, C_Redi, C_GM)

function with_tracers(tracers, closure::TwoDimensionalLeith{FT}) where FT
    C_Redi = tracer_diffusivities(tracers, closure.C_Redi)
    C_GM = tracer_diffusivities(tracers, closure.C_GM)
    return TwoDimensionalLeith{FT}(closure.C, C_Redi, C_GM)
end

@inline function abs²_∇h_ζ(i, j, k, grid, U)
    vxx = ℑyᵃᶜᵃ(i, j, k, grid, ∂²xᶜᵃᵃ, U.v)
    uyy = ℑxᶜᵃᵃ(i, j, k, grid, ∂²yᵃᶜᵃ, U.u)
    uxy = ℑyᵃᶜᵃ(i, j, k, grid, ∂xᶜᵃᵃ, ∂yᵃᶠᵃ, U.u)
    vxy = ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᵃᵃ, ∂yᵃᶜᵃ, U.v)

    return (vxx - uxy)^2 + (vxy - uyy)^2
end

const ArrayOrField = Union{AbstractArray, AbstractField}

@inline ψ²(i, j, k, grid, ψ::Function, args...) = ψ(i, j, k, grid, args...)^2
@inline ψ²(i, j, k, grid, ψ::ArrayOrField, args...) = ψ[i, j, k]^2

@inline function abs²_∇h_wz(i, j, k, grid, w)
    wxz² = ℑxᶜᵃᵃ(i, j, k, grid, ψ², ∂xᶠᵃᵃ, ∂zᵃᵃᶜ, w)
    wyz² = ℑyᵃᶜᵃ(i, j, k, grid, ψ², ∂yᵃᶠᵃ, ∂zᵃᵃᶜ, w)
    return wxz² + wyz²
end

@inline νᶜᶜᶜ(i, j, k, grid, clo::TwoDimensionalLeith{FT}, buoyancy, U, C) where FT =
    (clo.C * Δᶠ(i, j, k, grid, clo))^3 * sqrt(  abs²_∇h_ζ(i, j, k, grid, U)
                                              + abs²_∇h_wz(i, j, k, grid, U.w))

#####
##### Abstract Smagorinsky functionality
#####

# Components of the Redi rotation tensor

@inline function Redi_tensor_xz_fcc(i, j, k, grid::AbstractGrid{FT}, buoyancy, C) where FT
    bx = ∂x_b(i, j, k, grid, buoyancy, C)
    bz = ℑxzᶠᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, C)
    return ifelse(bx == 0 && bz == 0, zero(FT), - bx / bz)
end

@inline function Redi_tensor_xz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, C) where FT
    bx = ℑxzᶜᵃᶠ(i, j, k, grid, ∂x_b, buoyancy, C)
    bz = ∂z_b(i, j, k, grid, buoyancy, C)
    return ifelse(bx == 0 && bz == 0, zero(FT), - bx / bz)
end

@inline function Redi_tensor_yz_cfc(i, j, k, grid::AbstractGrid{FT}, buoyancy, C) where FT
    by = ∂y_b(i, j, k, grid, buoyancy, C)
    bz = ℑyzᵃᶠᶜ(i, j, k, grid, ∂z_b, buoyancy, C)
    return ifelse(by == 0 && bz == 0, zero(FT), - by / bz)
end

@inline function Redi_tensor_yz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, C) where FT
    by = ℑyzᵃᶜᶠ(i, j, k, grid, ∂y_b, buoyancy, C)
    bz = ∂z_b(i, j, k, grid, buoyancy, C)
    return ifelse(by == 0 && bz == 0, zero(FT), - by / bz)
end

@inline function Redi_tensor_zz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, C) where FT
    bx = ℑxzᶜᵃᶠ(i, j, k, grid, ∂x_b, buoyancy, C)
    by = ℑyzᵃᶜᶠ(i, j, k, grid, ∂y_b, buoyancy, C)
    bz = ∂z_b(i, j, k, grid, buoyancy, C)
    return ifelse(by == 0 && bx == 0 && bz == 0, zero(FT), (bx^2 + by^2) / bz^2)
end

# Diffusive fluxes for Leith diffusivities

@inline function diffusive_flux_x(i, j, k, grid, closure::TwoDimensionalLeith,
                                  c, ::Val{tracer_index}, clock, diffusivities, C, buoyancy) where tracer_index

    νₑ = diffusivities.νₑ

    @inbounds C_Redi = closure.C_Redi[tracer_index]
    @inbounds C_GM = closure.C_GM[tracer_index]

    νₑⁱʲᵏ = ℑxᶠᵃᵃ(i, j, k, grid, νₑ)

    ∂x_c = ∂xᶠᵃᵃ(i, j, k, grid, c)
    ∂z_c = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᵃᵃᶠ, c)

    R₁₃ = Redi_tensor_xz_fcc(i, j, k, grid, buoyancy, C)

    return - νₑⁱʲᵏ * (                 C_Redi * ∂x_c
                      + (C_Redi - C_GM) * R₁₃ * ∂z_c)
end

@inline function diffusive_flux_y(i, j, k, grid, closure::TwoDimensionalLeith,
                                  c, ::Val{tracer_index}, clock, diffusivities, C, buoyancy) where tracer_index

    νₑ = diffusivities.νₑ

    @inbounds C_Redi = closure.C_Redi[tracer_index]
    @inbounds C_GM = closure.C_GM[tracer_index]

    νₑⁱʲᵏ = ℑyᵃᶠᵃ(i, j, k, grid, νₑ)

    ∂y_c = ∂yᵃᶠᵃ(i, j, k, grid, c)
    ∂z_c = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᵃᵃᶠ, c)

    R₂₃ = Redi_tensor_yz_cfc(i, j, k, grid, buoyancy, C)
    return - νₑⁱʲᵏ * (                  C_Redi * ∂y_c
                             + (C_Redi - C_GM) * R₂₃ * ∂z_c)
end

@inline function diffusive_flux_z(i, j, k, grid, closure::TwoDimensionalLeith,
                                  c, ::Val{tracer_index}, clock, diffusivities, C, buoyancy) where tracer_index

    νₑ = diffusivities.νₑ

    @inbounds C_Redi = closure.C_Redi[tracer_index]
    @inbounds C_GM = closure.C_GM[tracer_index]

    νₑⁱʲᵏ = ℑzᵃᵃᶠ(i, j, k, grid, νₑ)

    ∂x_c = ℑxzᶜᵃᶠ(i, j, k, grid, ∂xᶠᵃᵃ, c)
    ∂y_c = ℑyzᵃᶜᶠ(i, j, k, grid, ∂yᵃᶠᵃ, c)
    ∂z_c = ∂zᵃᵃᶠ(i, j, k, grid, c)

    R₃₁ = Redi_tensor_xz_ccf(i, j, k, grid, buoyancy, C)
    R₃₂ = Redi_tensor_yz_ccf(i, j, k, grid, buoyancy, C)
    R₃₃ = Redi_tensor_zz_ccf(i, j, k, grid, buoyancy, C)

    return - νₑⁱʲᵏ * (
          (C_Redi + C_GM) * R₃₁ * ∂x_c
        + (C_Redi + C_GM) * R₃₂ * ∂y_c
                 + C_Redi * R₃₃ * ∂z_c)
end

const L2D = TwoDimensionalLeith

@inline viscous_flux_ux(i, j, k, grid, closure::L2D, clock, U, K) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, closure, K.νₑ, Σ₁₁, U.u, U.v, U.w)
@inline viscous_flux_uy(i, j, k, grid, closure::L2D, clock, U, K) = - 2 * ν_σᶠᶠᶜ(i, j, k, grid, closure, K.νₑ, Σ₁₂, U.u, U.v, U.w)
@inline viscous_flux_uz(i, j, k, grid, closure::L2D, clock, U, K) = - 2 * ν_σᶠᶜᶠ(i, j, k, grid, closure, K.νₑ, Σ₁₃, U.u, U.v, U.w)
                                                            
@inline viscous_flux_vx(i, j, k, grid, closure::L2D, clock, U, K) = - 2 * ν_σᶠᶠᶜ(i, j, k, grid, closure, K.νₑ, Σ₂₁, U.u, U.v, U.w)
@inline viscous_flux_vy(i, j, k, grid, closure::L2D, clock, U, K) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, closure, K.νₑ, Σ₂₂, U.u, U.v, U.w)
@inline viscous_flux_vz(i, j, k, grid, closure::L2D, clock, U, K) = - 2 * ν_σᶜᶠᶠ(i, j, k, grid, closure, K.νₑ, Σ₂₃, U.u, U.v, U.w)
                                                            
@inline viscous_flux_wx(i, j, k, grid, closure::L2D, clock, U, K) = - 2 * ν_σᶠᶜᶠ(i, j, k, grid, closure, K.νₑ, Σ₃₁, U.u, U.v, U.w)
@inline viscous_flux_wy(i, j, k, grid, closure::L2D, clock, U, K) = - 2 * ν_σᶜᶠᶠ(i, j, k, grid, closure, K.νₑ, Σ₃₂, U.u, U.v, U.w)
@inline viscous_flux_wz(i, j, k, grid, closure::L2D, clock, U, K) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, closure, K.νₑ, Σ₃₃, U.u, U.v, U.w)

function calculate_diffusivities!(K, arch, grid, closure::TwoDimensionalLeith, buoyancy, U, C)
    event = launch!(arch, grid, :xyz, calculate_nonlinear_viscosity!, K.νₑ, grid, closure, buoyancy, U, C,
                    dependencies=Event(device(arch)))

    wait(device(arch), event)

    return nothing
end

"Return the filter width for a Leith Diffusivity on a regular rectilinear grid."
@inline Δᶠ(i, j, k, grid::RegularRectilinearGrid, ::TwoDimensionalLeith) = sqrt(grid.Δx * grid.Δy)

function DiffusivityFields(arch, grid, tracer_names, bcs, ::L2D)
    default_eddy_viscosity_bcs = (; νₑ = FieldBoundaryConditions(grid, (Center, Center, Center)))
    bcs = merge(default_eddy_viscosity_bcs, bcs)
    νₑ = CenterField(arch, grid, bcs.νₑ)
    return (; νₑ)
end
