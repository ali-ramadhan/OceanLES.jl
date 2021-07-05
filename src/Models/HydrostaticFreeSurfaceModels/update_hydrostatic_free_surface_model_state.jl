using Oceananigans.Architectures
using Oceananigans.BoundaryConditions
using Oceananigans.TurbulenceClosures: calculate_diffusivities!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Models.IncompressibleModels: update_hydrostatic_pressure!

import Oceananigans.TimeSteppers: update_state!

"""
    update_state!(model::HydrostaticFreeSurfaceModel)

Update peripheral aspects of the model (auxiliary fields, halo regions, diffusivities,
hydrostatic pressure) to the current model state.
"""
function update_state!(model::HydrostaticFreeSurfaceModel)

    # Mask immersed fields
    masking_events = Tuple(mask_immersed_field!(field) for field in merge(model.auxiliary_fields, prognostic_fields(model)))

    wait(device(model.architecture), MultiEvent(masking_events))

    # Fill halos for velocities and tracers. On the CubedSphere, the halo filling for velocity fields is wrong.
    fill_halo_regions!(prognostic_fields(model), model.clock, fields(model))

    # This _refills_ the halos for horizontal velocity fields when grid::ConformalCubedSphereGrid
    # For every other type of grid, fill_horizontal_velocity_halos! does nothing.
    fill_horizontal_velocity_halos!(model.velocities.u, model.velocities.v)

    compute_w_from_continuity!(model)

    fill_halo_regions!(model.velocities.w, model.clock, fields(model))

    # Compute auxiliary fields
    for aux_field in model.auxiliary_fields
        compute!(aux_field)
    end

    # Calculate diffusivities
    calculate_diffusivities!(model.diffusivities, model.architecture, model.grid, model.closure,
                             model.buoyancy, model.velocities, model.tracers)

    fill_halo_regions!(model.diffusivities, model.clock, fields(model))

    # Calculate hydrostatic pressure
    pressure_calculation = launch!(model.architecture, model.grid, Val(:xy), update_hydrostatic_pressure!,
                                   model.pressure.pHY′, model.grid, model.buoyancy, model.tracers,
                                   dependencies=Event(device(model.architecture)))

    # Fill halo regions for pressure
    wait(device(model.architecture), pressure_calculation)

    fill_halo_regions!(model.pressure.pHY′)

    return nothing
end
