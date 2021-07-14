module ForcedFlowFixedSlip

using Printf

using Oceananigans

# Functions that define the forced flow problem

 ξ(t) = 1 + sin(t^2)
ξ′(t) = 2t * cos(t^2)

 f(x, t) =   cos(x - ξ(t))
fₓ(x, t) = - sin(x - ξ(t))

 g(y) = y^3 - y^2
g′(y) = 3y^2 - 2y

H₁(y) = y^4 / 4 - y^3 / 3 - 3y^2 + 2y
H₂(y) = 3y^4 - 4y^3 + 2y^2
H₃(y) = y^4 / 4 - y^3 / 3 - 6y^2 + 4y

Fᵘ(x, y, t) = ξ′(t) * fₓ(x, t) * H₁(y) + f(x, t) * fₓ(x, t) * H₂(y) - f(x, t) * H₃(y)

Fᵛ(x, y, t) = 3y^5 - 5y^4 + 2y^3

u(x, y, t) = f(x, t) * g′(y)
v(x, y, t) = - fₓ(x, t) * g(y)

function print_progress(simulation)
    model = simulation.model
    i, t = model.clock.iteration, model.clock.time
    progress = 100 * (i / simulation.stop_iteration)
    @info @sprintf("[%05.2f%%] iteration: %d, time: %.5e", progress, i, t)
    return nothing
end

const DATA_DIR = joinpath(@__DIR__, "..", "data")

function setup_xy_simulation(; Nx, Δt, stop_iteration, architecture=CPU(), dir=DATA_DIR)

    u_forcing(x, y, z, t) = Fᵘ(x, y, t)
    v_forcing(x, y, z, t) = Fᵛ(x, y, t)

    grid = RegularRectilinearGrid(size=(Nx, Nx, 1), x=(0, 2π), y=(0, 1), z=(0, 1),
                                topology=(Periodic, Bounded, Bounded))

    # "Fixed slip" boundary conditions (eg, no-slip on south wall, finite slip on north wall)."
    u_bcs = FieldBoundaryConditions(north = ValueBoundaryCondition((x, y, t) -> f(x, t)),
                                    south = ValueBoundaryCondition(0))

    model = IncompressibleModel(architecture = CPU(),
                                grid = grid,
                                coriolis = nothing,
                                buoyancy = nothing,
                                tracers = nothing,
                                closure = IsotropicDiffusivity(ν=1),
                                boundary_conditions = (u=u_bcs,),
                                forcing = (u=u_forcing, v=v_forcing))

    set!(model, u = (x, y, z) -> u(x, y, 0),
                v = (x, y, z) -> v(x, y, 0))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, progress=print_progress, iteration_interval=20)

    outputs = Dict()
    pressure_output = model -> parent(model.pressures.pHY′) .+ parent(model.pressures.pNHS)
    outputs = merge((p=pressure_output,), model.velocities)

    simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                          dir = dir, force = true,
                                                          prefix = @sprintf("forced_fixed_slip_xy_Nx%d_Δt%.1e", Nx, Δt),
                                                          field_slicer = nothing,
                                                          schedule = TimeInterval(stop_iteration * Δt / 2))

    return simulation
end

function setup_and_run_xy(args...; kwargs...)
    simulation = setup_xy_simulation(args...; kwargs...)
    @info "Running forced flow fixed slip simulation with Nx = Ny = $(kwargs[:Nx]), Δt = $(kwargs[:Δt])"
    @time run!(simulation)
    return simulation
end

function setup_xz_simulation(; Nx, Δt, stop_iteration, architecture=CPU(), dir=DATA_DIR)

    u_forcing(x, y, z, t) = Fᵘ(x, z, t)
    w_forcing(x, y, z, t) = Fᵛ(x, z, t)

    grid = RegularRectilinearGrid(size=(Nx, 1, Nx), x=(0, 2π), y=(0, 1), z=(0, 1),
                                topology=(Periodic, Bounded, Bounded))

    # "Fixed slip" boundary conditions (eg, no-slip on bottom and finite slip on top)."
    u_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition((x, z, t) -> f(x, t)),
                                    bottom = ValueBoundaryCondition(0))

    model = IncompressibleModel(architecture = CPU(),
                                grid = grid,
                                coriolis = nothing,
                                buoyancy = nothing,
                                tracers = nothing,
                                closure = IsotropicDiffusivity(ν=1),
                                boundary_conditions = (u=u_bcs,),
                                forcing = (u=u_forcing, w=w_forcing))

    set!(model, u = (x, y, z) -> u(x, z, 0),
                w = (x, y, z) -> v(x, z, 0))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, progress=print_progress, iteration_interval=20)

    outputs = Dict()
    pressure_output = model -> parent(model.pressures.pHY′) .+ parent(model.pressures.pNHS)
    outputs = merge((p=pressure_output,), model.velocities)

    simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                          dir = dir, force = true,
                                                          prefix = @sprintf("forced_fixed_slip_xz_Nx%d_Δt%.1e", Nx, Δt),
                                                          field_slicer = nothing,
                                                          schedule = TimeInterval(stop_iteration * Δt / 2))

    return simulation
end

function setup_and_run_xz(args...; kwargs...)
    simulation = setup_xz_simulation(args...; kwargs...)
    @info "Running forced flow fixed slip simulation with Nx = Nz = $(kwargs[:Nx]), Δt = $(kwargs[:Δt])"
    @time run!(simulation)
    return simulation
end

end # module
