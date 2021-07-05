module BoundaryConditions

export
    BCType, Flux, Gradient, Value, NormalFlow,

    BoundaryCondition, bctype, getbc, setbc!,

    PeriodicBoundaryCondition, NormalFlowBoundaryCondition, NoFluxBoundaryCondition,
    FluxBoundaryCondition, ValueBoundaryCondition, GradientBoundaryCondition,

    CoordinateBoundaryConditions,

    FieldBoundaryConditions, UVelocityBoundaryConditions, VVelocityBoundaryConditions,
    WVelocityBoundaryConditions, TracerBoundaryConditions, PressureBoundaryConditions,

    DiffusivityBoundaryConditions,

    apply_x_bcs!, apply_y_bcs!, apply_z_bcs!,

    fill_halo_regions!

using CUDA
using KernelAbstractions

using Oceananigans.Architectures: device
using Oceananigans.Utils: work_layout, launch!
using Oceananigans.Operators: Δx, Δy, Δz, Ax, Ay, Az, volume
using Oceananigans.Grids

include("boundary_condition_types.jl")
include("boundary_condition.jl")
include("discrete_boundary_function.jl")
include("continuous_boundary_function.jl")
include("coordinate_boundary_conditions.jl")
include("field_boundary_conditions.jl")
include("show_boundary_conditions.jl")

include("fill_halo_regions.jl")
include("fill_halo_regions_value_gradient.jl")
include("fill_halo_regions_normal_flow.jl")
include("fill_halo_regions_periodic.jl")
include("fill_halo_regions_flux.jl")
include("fill_halo_regions_nothing.jl")

include("apply_flux_bcs.jl")

end
