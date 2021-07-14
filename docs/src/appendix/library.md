# Library

Documenting the public user interface.

## Advection

```@autodocs
Modules = [Oceananigans.Advection]
Private = false
Pages   = [
    "Advection/Advection.jl",
    "Advection/tracer_advection_operators.jl",
    "Advection/momentum_advection_operators.jl",
    "Advection/centered_second_order.jl",
    "Advection/centered_fourth_order.jl",
    "Advection/upwind_biased_third_order.jl",
    "Advection/upwind_biased_fifth_order.jl",
    "Advection/weno_fifth_order.jl"
]
```

## Architectures

```@autodocs
Modules = [Oceananigans.Architectures]
Private = false
Pages   = ["Architectures.jl"]
```

## Boundary conditions

```@autodocs
Modules = [Oceananigans.BoundaryConditions]
Private = false
Pages   = [
    "BoundaryConditions/BoundaryConditions.jl",
    "BoundaryConditions/boundary_condition_classifications.jl",
    "BoundaryConditions/boundary_condition.jl",
    "BoundaryConditions/discrete_boundary_function.jl",
    "BoundaryConditions/continuous_boundary_function.jl",
    "BoundaryConditions/field_boundary_conditions.jl",
    "BoundaryConditions/show_boundary_conditions.jl",
    "BoundaryConditions/fill_halo_regions.jl",
    "BoundaryConditions/fill_halo_regions_value_gradient.jl",
    "BoundaryConditions/fill_halo_regions_open.jl",
    "BoundaryConditions/fill_halo_regions_periodic.jl",
    "BoundaryConditions/fill_halo_regions_flux.jl",
    "BoundaryConditions/fill_halo_regions_nothing.jl",
    "BoundaryConditions/apply_flux_bcs.jl",
]
```

## BuoyancyModels

```@autodocs
Modules = [Oceananigans.BuoyancyModels]
Private = false
Pages   = [
    "BuoyancyModels/no_buoyancy.jl",
    "BuoyancyModels/buoyancy_tracer.jl",
    "BuoyancyModels/seawater_buoyancy.jl",
    "BuoyancyModels/BuoyancyModels.jl",
    "BuoyancyModels/linear_equation_of_state.jl",
    "BuoyancyModels/nonlinear_equation_of_state.jl",
    "BuoyancyModels/roquet_idealized_nonlinear_eos.jl",
    "BuoyancyModels/show_buoyancy.jl",
    "BuoyancyModels/buoyancy_utils.jl"
]
```

## Coriolis

```@autodocs
Modules = [Oceananigans.Coriolis]
Private = false
Pages   = [
    "Coriolis/Coriolis.jl",
    "Coriolis/no_rotation.jl",
    "Coriolis/f_plane.jl",
    "Coriolis/beta_plane.jl",
    "Coriolis/non_traditional_f_plane.jl",
    "Coriolis/non_traditional_beta_plane.jl"
]
```

## Diagnostics

```@autodocs
Modules = [Oceananigans.Diagnostics]
Private = false
Pages   = [
    "Diagnostics/Diagnostics.jl",
    "Diagnostics/diagnostics_kernels.jl",
    "Diagnostics/average.jl",
    "Diagnostics/time_series.jl",
    "Diagnostics/cfl.jl",
    "Diagnostics/field_maximum.jl",
    "Diagnostics/nan_checker.jl"
]
```

## Fields

```@autodocs
Modules = [Oceananigans.Fields]
Private = false
Pages   = [
    "Fields/Fields.jl",
    "Fields/field.jl",
    "Fields/set!.jl",
    "Fields/show_fields.jl",
]
```

## Forcings

```@autodocs
Modules = [Oceananigans.Forcings]
Private = false
Pages   = [
    "Forcings/Forcings.jl",
    "Forcings/continuous_forcing.jl",
    "Forcings/discrete_forcing.jl",
    "Forcings/forcing.jl",
    "Forcings/model_forcing.jl",
    "Forcings/relaxation.jl"
]
```

## Grids

```@autodocs
Modules = [Oceananigans.Grids]
Private = false
Pages   = [
    "Grids/Grids.jl",
    "Grids/grid_utils.jl",
    "Grids/regular_rectilinear_grid.jl",
    "Grids/vertically_stretched_rectilinear_grid.jl"
]
```

## Lagrangian particle tracking

```@autodocs
Modules = [Oceananigans.LagrangianParticleTracking]
Private = false
Pages   = [
    "LagrangianParticleTracking/LagrangianParticleTracking.jl",
    "LagrangianParticleTracking/advect_particles.jl"
]
```

## Logger

```@autodocs
Modules = [Oceananigans.Logger]
Private = false
Pages   = ["Logger.jl"]
```

## Models

```@autodocs
Modules = [Oceananigans.Models]
Private = false
Pages   = [
    "Models/Models.jl",
    "Models/IncompressibleModels/incompressible_model.jl",
    "Models/HydrostaticFreeSurfaceModels/hydrostatic_free_surface_model.jl",
    "Models/ShallowWaterModels/shallow_water_model.jl"
]
```

## Output writers

```@autodocs
Modules = [Oceananigans.OutputWriters]
Private = false
Pages   = [
    "OutputWriters/OutputWriters.jl",
    "OutputWriters/output_writer_utils.jl",
    "OutputWriters/jld2_output_writer.jl",
    "OutputWriters/netcdf_output_writer.jl",
    "OutputWriters/windowed_time_average.jl",
    "OutputWriters/checkpointer.jl"
]
```

## Time steppers

```@autodocs
Modules = [Oceananigans.TimeSteppers]
Private = false
Pages   = [
    "TimeSteppers/TimeSteppers.jl",
    "TimeSteppers/clock.jl",
    "TimeSteppers/quasi_adams_bashforth_2.jl",
    "TimeSteppers/runge_kutta_3.jl
]
```

## Simulations

```@autodocs
Modules = [Oceananigans.Simulations]
Private = false
Pages   = [
    "Simulations/Simulations.jl",
    "Simulations/time_step_wizard.jl",
    "Simulations/simulation.jl",
    "Simulations/run.jl"
]
```

## Tubrulence closures

```@autodocs
Modules = [Oceananigans.TurbulenceClosures]
Private = false
Pages   = [
    "TurbulenceClosures/TurbulenceClosures.jl",
    "TurbulenceClosures/turbulence_closure_utils.jl",
    "TurbulenceClosures/closure_operators.jl",
    "TurbulenceClosures/viscous_dissipation_operators.jl",
    "TurbulenceClosures/diffusion_operators.jl",
    "TurbulenceClosures/velocity_tracer_gradients.jl",
    "TurbulenceClosures/closure_tuples.jl",
    "TurbulenceClosures/turbulence_closure_diagnostics.jl",
    "TurbulenceClosures/turbulence_closure_implementations/anisotropic_biharmonic_diffusivity.jl",
    "TurbulenceClosures/turbulence_closure_implementations/smagorinsky_lilly.jl",
    "TurbulenceClosures/turbulence_closure_implementations/isotropic_diffusivity.jl",
    "TurbulenceClosures/turbulence_closure_implementations/anisotropic_diffusivity.jl",
    "TurbulenceClosures/turbulence_closure_implementations/anisotropic_minimum_dissipation.jl",
    "TurbulenceClosures/turbulence_closure_implementations/leith_enstrophy_diffusivity.jl",
    "TurbulenceClosures/turbulence_closure_implementations/tke_based_vertical_diffusivity.jl"
]
```

## Utilities

```@autodocs
Modules = [Oceananigans.Utils]
Private = false
Pages   = [
    "Utils/Utils.jl",
    "Utils/kernel_launcing.jl",
    "Utils/pretty_time.jl",
    "Utils/pretty_filesize.jl",
    "Utils/time_step_wizard.jl",
    "Utils/tuple_utils.jl",
    "Utils/ordered_dict_show.jl",
    "Utils/cell_advection_timescale.jl",
    "Utils/output_writer_diagnostic_utils.jl",
    "Utils/with_tracers.jl",
    "Utils/schedules.jl"
]
```

## Abstract operations

```@autodocs
Modules = [Oceananigans, Oceananigans.AbstractOperations]
Private = false
Pages   = [
    "AbstractOperations/AbstractOperations.jl",
    "AbstractOperations/unary_operations.jl",
    "AbstractOperations/binary_operations.jl",
    "AbstractOperations/multiary_operations.jl",
    "AbstractOperations/derivatives.jl",
    "AbstractOperations/function_fields.jl",
    "AbstractOperations/computations.jl",
    "AbstractOperations/interpolation_utils.jl",
    "AbstractOperations/show_abstract_operations.jl",
    "AbstractOperations/grid_validation.jl"
]
```
