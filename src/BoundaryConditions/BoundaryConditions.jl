module BoundaryConditions

export
    BCType, Flux, Gradient, Value, Open,
    BoundaryCondition, getbc, setbc!,
    PeriodicBoundaryCondition, OpenBoundaryCondition, NoFluxBoundaryCondition,
    FluxBoundaryCondition, ValueBoundaryCondition, GradientBoundaryCondition,
    FieldBoundaryConditions,
    apply_x_bcs!, apply_y_bcs!, apply_z_bcs!,
    fill_halo_regions!

using CUDA
using KernelAbstractions

using Oceananigans.Architectures: device
using Oceananigans.Utils: work_layout, launch!
using Oceananigans.Operators: Δx, Δy, Ax, Ay, Az, volume, ΔzF
using Oceananigans.Grids

include("boundary_condition_classifications.jl")
include("boundary_condition.jl")
include("discrete_boundary_function.jl")
include("continuous_boundary_function.jl")
include("field_boundary_conditions.jl")
include("show_boundary_conditions.jl")

include("fill_halo_regions.jl")
include("fill_halo_regions_value_gradient.jl")
include("fill_halo_regions_open.jl")
include("fill_halo_regions_periodic.jl")
include("fill_halo_regions_flux.jl")
include("fill_halo_regions_nothing.jl")

include("apply_flux_bcs.jl")

end
