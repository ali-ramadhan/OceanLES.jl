using Oceananigans.Grids: AbstractGrid
using Oceananigans.Architectures: device
using Oceananigans.Operators: ∂xᶠᵃᵃ, ∂yᵃᶠᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Solvers: solve!
using Oceananigans.Fields

using Adapt
using KernelAbstractions: NoneEvent

struct ImplicitFreeSurface{E, G, B, V, R, I, S}
    η :: E
    gravitational_acceleration :: G
    barotropic_volume_flux :: B
    vertically_integrated_lateral_face_areas :: V
    implicit_step_right_hand_side :: R
    implicit_step_solver :: I
    solver_settings :: S
end

# User interface to ImplicitFreeSurface
ImplicitFreeSurface(; gravitational_acceleration=g_Earth, solver_settings...) =
    ImplicitFreeSurface(nothing, gravitational_acceleration, nothing, nothing, nothing, nothing, solver_settings)

Adapt.adapt_structure(to, free_surface::ImplicitFreeSurface) =
    ImplicitFreeSurface(Adapt.adapt(to, free_surface.η), free_surface.gravitational_acceleration,
                        nothing, nothing, nothing, nothing, nothing)

# Internal function for HydrostaticFreeSurfaceModel
function FreeSurface(free_surface::ImplicitFreeSurface{Nothing}, velocities, arch, grid)
    η = FreeSurfaceDisplacementField(velocities, free_surface, arch, grid)
    g = convert(eltype(grid), free_surface.gravitational_acceleration)

    # Initialize barotropic volume fluxes
    barotropic_x_volume_flux = ReducedField(Face, Center, Nothing, arch, grid; dims=3)
    barotropic_y_volume_flux = ReducedField(Center, Face, Nothing, arch, grid; dims=3)
    barotropic_volume_flux = (u=barotropic_x_volume_flux, v=barotropic_y_volume_flux)

    # Initialize vertically integrated lateral face areas
    ∫ᶻ_Axᶠᶜᶜ = ReducedField(Face, Center, Nothing, arch, grid; dims=3)
    ∫ᶻ_Ayᶜᶠᶜ = ReducedField(Center, Face, Nothing, arch, grid; dims=3)

    vertically_integrated_lateral_face_areas = (xᶠᶜᶜ = ∫ᶻ_Axᶠᶜᶜ, yᶜᶠᶜ = ∫ᶻ_Ayᶜᶠᶜ)

    compute_vertically_integrated_lateral_face_areas!(vertically_integrated_lateral_face_areas, grid, arch)

    implicit_step_solver = PreconditionedConjugateGradientSolver(implicit_free_surface_linear_operation!,
                                                                 template_field = η,
                                                                 maximum_iterations = grid.Nx * grid.Ny;
                                                                 free_surface.solver_settings...)

    implicit_step_right_hand_side = ReducedField(Center, Center, Nothing, arch, grid; dims=3)

    return ImplicitFreeSurface(η,
                               g,
                               barotropic_volume_flux,
                               vertically_integrated_lateral_face_areas,
                               implicit_step_right_hand_side,
                               implicit_step_solver,
                               free_surface.solver_settings)
end

@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::ImplicitFreeSurface) = 0
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::ImplicitFreeSurface) = 0

"""
Implicitly step forward η.
"""
ab2_step_free_surface!(free_surface::ImplicitFreeSurface, model, Δt, χ, velocities_update) =
    implicit_free_surface_step!(free_surface::ImplicitFreeSurface, model, Δt, χ, velocities_update)

function implicit_free_surface_step!(free_surface::ImplicitFreeSurface, model, Δt, χ, velocities_update)

    η = free_surface.η
    g = free_surface.gravitational_acceleration
    rhs = free_surface.implicit_step_right_hand_side
    ∫ᶻ_Q = free_surface.barotropic_volume_flux
    ∫ᶻ_A = free_surface.vertically_integrated_lateral_face_areas

    #=
    # Take an explicit step first to produce an improved initial guess for η for the iterative solver.
    event = explicit_ab2_step_free_surface!(free_surface, model, Δt, χ)
    wait(device(model.architecture), event)
    =#

    fill_halo_regions!(η)

    compute_vertically_integrated_volume_flux!(∫ᶻ_Q, model, velocities_update)

    compute_implicit_free_surface_right_hand_side!(rhs, model, g, Δt, ∫ᶻ_Q, η)

    fill_halo_regions!(rhs)

    # solve!(x, solver, b, args...) solves A*x = b for x.
    solve!(η, free_surface.implicit_step_solver, rhs, ∫ᶻ_A.xᶠᶜᶜ, ∫ᶻ_A.yᶜᶠᶜ, g, Δt)

    return NoneEvent()
end

function compute_implicit_free_surface_right_hand_side!(rhs, model, g, Δt, ∫ᶻ_Q, η)

    event = launch!(model.architecture,
                    model.grid,
                    :xy,
                    implicit_free_surface_right_hand_side!,
                    rhs,
                    model.grid,
                    g,
                    Δt,
                    ∫ᶻ_Q,
                    η,
                    dependencies=Event(device(model.architecture)))


    wait(device(model.architecture), event)

    fill_halo_regions!(rhs)

    return nothing
end

""" Compute the divergence of fluxes Qu and Qv. """
@inline flux_div_xyᶜᶜᵃ(i, j, k, grid, Qu, Qv) = δxᶜᵃᵃ(i, j, k, grid, Qu) + δyᵃᶜᵃ(i, j, k, grid, Qv)

@kernel function implicit_free_surface_right_hand_side!(rhs, grid, g, Δt, ∫ᶻ_Q, η)
    i, j = @index(Global, NTuple)
    @inbounds rhs[i, j, 1] = - Azᶜᶜᵃ(i, j, 1, grid) * η[i, j, 1] / (g * Δt^2) +
                               flux_div_xyᶜᶜᵃ(i, j, 1, grid, ∫ᶻ_Q.u, ∫ᶻ_Q.v) / (g * Δt)
end
