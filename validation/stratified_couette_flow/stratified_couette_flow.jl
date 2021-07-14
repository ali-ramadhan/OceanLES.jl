using Statistics, Printf

using Oceananigans
using Oceananigans.Fields
using Oceananigans.TurbulenceClosures
using Oceananigans.OutputWriters
using Oceananigans.Diagnostics
using Oceananigans.Utils

""" Friction velocity. See equation (16) of Vreugdenhil & Taylor (2018). """
function uτ(model, Uavg, U_wall)
    Nz, Hz, Δz = model.grid.Nz, model.grid.Hz, model.grid.Δz
    ν = model.closure.ν

    compute!(Uavg)
    U = Array(interior(Uavg))  # Exclude average of halo region.

    # Use a finite difference to calculate dU/dz at the top and bottomtom walls.
    # The distance between the center of the cell adjacent to the wall and the
    # wall itself is Δz/2.
    uτ²_top    = ν * abs(U[1] - U_wall)    / (Δz/2)  # Top wall    where u = +U_wall
    uτ²_bottom = ν * abs(-U_wall  - U[Nz]) / (Δz/2)  # Bottom wall where u = -U_wall

    uτ_top, uτ_bottom = √uτ²_top, √uτ²_bottom

    return uτ_top, uτ_bottom
end

""" Heat flux at the wall. See equation (16) of Vreugdenhil & Taylor (2018). """
function q_wall(model, Tavg, Θ_wall)
    Nz, Hz, Δz = model.grid.Nz, model.grid.Hz, model.grid.Δz
    κ = model.closure.κ.T

    compute!(Tavg)
    Θ = Array(interior(Tavg)) # Exclude average of halo region.

    # Use a finite difference to calculate dθ/dz at the top and bottomtom walls.
    # The distance between the center of the cell adjacent to the wall and the
    # wall itself is Δz/2.
    q_wall_top    = κ * abs(Θ[1] - Θ_wall)   / (Δz/2)  # Top wall    where Θ = +Θ_wall
    q_wall_bottom = κ * abs(-Θ_wall - Θ[Nz]) / (Δz/2)  # Bottom wall where Θ = -Θ_wall

    return q_wall_top, q_wall_bottom
end

struct FrictionReynoldsNumber{H, U}
    Uavg :: H
    U_wall :: U
end

struct NusseltNumber{H, T}
    Tavg :: H
    Θ_wall :: T
end

""" Friction Reynolds number. See equation (20) of Vreugdenhil & Taylor (2018). """
function (Reτ::FrictionReynoldsNumber)(model)
    ν = model.closure.ν
    h = model.grid.Lz / 2
    uτ_top, uτ_bottom = uτ(model, Reτ.Uavg, Reτ.U_wall)

    return h * uτ_top / ν, h * uτ_bottom / ν
end

""" Nusselt number. See equation (20) of Vreugdenhil & Taylor (2018). """
function (Nu::NusseltNumber)(model)
    κ = model.closure.κ.T
    h = model.grid.Lz / 2

    q_wall_top, q_wall_bottom = q_wall(model, Nu.Tavg, Nu.Θ_wall)

    return (q_wall_top * h)/(κ * Nu.Θ_wall), (q_wall_bottom * h)/(κ * Nu.Θ_wall)
end

"""
    simulate_stratified_couette_flow(; Nxy, Nz, h=1, U_wall=1, Re=4250, Pr=0.7,
                                     Ri, Ni=10, end_time=1000)

Simulate stratified plane Couette flow with `Nxy` grid cells in each horizontal
direction, `Nz` grid cells in the vertical, in a domain of size (4πh, 2πh, 2h),
with wall velocities of `U_wall` at the top and -`U_wall` at the bottom, at a Reynolds
number `Re, Prandtl number `Pr`, and Richardson number `Ri`.

`Ni` is the number of "intermediate" time steps taken at a time before printing a progress
statement and updating the time step.
"""
function simulate_stratified_couette_flow(; Nxy, Nz, arch=GPU(), h=1, U_wall=1,
                                          Re=4250, Pr=0.7, Ri, Ni=10, end_time=1000)
    #####
    ##### Computed parameters
    #####

         ν = U_wall * h / Re    # From Re = U_wall h / ν
    Θ_wall = Ri * U_wall^2 / h  # From Ri = L Θ_wall / U_wall²
         κ = ν / Pr             # From Pr = ν / κ

    #####
    ##### Impose boundary conditions
    #####

    grid = RegularRectilinearGrid(size = (Nxy, Nxy, Nz), extent = (4π*h, 2π*h, 2h))

    Tbcs = FieldBoundaryConditions(top = BoundaryCondition(Value,  Θ_wall),
                                   bottom = BoundaryCondition(Value, -Θ_wall))

    ubcs = FieldBoundaryConditions(top = BoundaryCondition(Value,  U_wall),
                                   bottom = BoundaryCondition(Value, -U_wall))

    vbcs = FieldBoundaryConditions(top = BoundaryCondition(Value, 0),
                                   bottom = BoundaryCondition(Value, 0))

    #####
    ##### Non-dimensional model setup
    #####

    model = IncompressibleModel(
               architecture = arch,
                       grid = grid,
                   buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=1.0, β=0.0)),
                    closure = AnisotropicMinimumDissipation(ν=ν, κ=κ),
        boundary_conditions = (u=ubcs, v=vbcs, T=Tbcs)
    )

    #####
    ##### Set initial conditions
    #####

    # Add a bit of surface-concentrated noise to the initial condition
    ε(σ, z) = σ * randn() * z/model.grid.Lz * (1 + z/model.grid.Lz)

    # We add a sinusoidal initial condition to u to encourage instability.
    T₀(x, y, z) = 2Θ_wall * (1/2 + z/model.grid.Lz) * (1 + ε(5e-1, z))
    u₀(x, y, z) = 2U_wall * (1/2 + z/model.grid.Lz) * (1 + ε(5e-1, z)) * (1 + 0.5*sin(4π/model.grid.Lx * x))
    v₀(x, y, z) = ε(5e-1, z)
    w₀(x, y, z) = ε(5e-1, z)

    set!(model, u=u₀, v=v₀, w=w₀, T=T₀)

    #####
    ##### Print simulation banner
    #####

    @printf(
        """
        Simulating stratified plane Couette flow

                N : %d, %d, %d
                L : %.3g, %.3g, %.3g
               Re : %.3f
               Ri : %.3f
               Pr : %.3f
                ν : %.3g
                κ : %.3g
           U_wall : %.3f
           Θ_wall : %.3f

        """, model.grid.Nx, model.grid.Ny, model.grid.Nz,
             model.grid.Lx, model.grid.Ly, model.grid.Lz,
             Re, Ri, Pr, ν, κ, U_wall, Θ_wall)

    #####
    ##### Set up field output writer
    #####

    base_dir = @sprintf("stratified_couette_flow_data_Nxy%d_Nz%d_Ri%.2f", Nxy, Nz, Ri)
    prefix = @sprintf("stratified_couette_flow_Nxy%d_Nz%d_Ri%.2f", Nxy, Nz, Ri)

    function init_save_parameters_and_bcs(file, model)
        file["parameters/reynolds_number"] = Re
        file["parameters/richardson_number"] = Ri
        file["parameters/prandtl_number"] = Pr
        file["parameters/viscosity"] = ν
        file["parameters/diffusivity"] = κ
        file["parameters/wall_velocity"] = U_wall
        file["parameters/wall_temperature"] = Θ_wall
    end

    fields = Dict(
        :u => model -> Array(model.velocities.u.data.parent),
        :v => model -> Array(model.velocities.v.data.parent),
        :w => model -> Array(model.velocities.w.data.parent),
        :T => model -> Array(model.tracers.T.data.parent),
   :kappaT => model -> Array(model.diffusivities.κₑ.T.data.parent),
       :nu => model -> Array(model.diffusivities.νₑ.data.parent))

    field_writer =
        JLD2OutputWriter(model, fields, dir=base_dir, prefix=prefix * "_fields",
                         init=init_save_parameters_and_bcs, schedule=TimeInterval(10),
                         force=true, verbose=true)

    #####
    ##### Set up profile output writer
    #####

    Uavg = AveragedField(model.velocities.u,       dims=(1, 2))
    Vavg = AveragedField(model.velocities.v,       dims=(1, 2))
    Wavg = AveragedField(model.velocities.w,       dims=(1, 2))
    Tavg = AveragedField(model.tracers.T,          dims=(1, 2))
    νavg = AveragedField(model.diffusivities.νₑ,   dims=(1, 2))
    κavg = AveragedField(model.diffusivities.κₑ.T, dims=(1, 2))

    profiles = Dict(
         :u => Uavg,
         :v => Vavg,
         :w => Wavg,
         :T => Tavg,
        :nu => νavg,
    :kappaT => κavg)

    profile_writer =
        JLD2OutputWriter(model, profiles, dir=base_dir, prefix=prefix * "_profiles",
                         init=init_save_parameters_and_bcs, schedule=TimeInterval(1),
                         force=true, verbose=true)

    #####
    ##### Set up statistic output writer
    #####

    Reτ = FrictionReynoldsNumber(Uavg, U_wall)
     Nu = NusseltNumber(Tavg, Θ_wall)

    statistics = Dict(
        :Re_tau => model -> Reτ(model),
        :Nu     => model -> Nu(model))

    statistics_writer =
        JLD2OutputWriter(model, statistics, dir=base_dir, prefix=prefix * "_statistics",
                         init=init_save_parameters_and_bcs, schedule=TimeInterval(1/2),
                         force=true, verbose=true)

    #####
    ##### Time stepping
    #####

    wizard = TimeStepWizard(cfl=0.02, Δt=0.0001, max_change=1.1, max_Δt=0.02)

    # We will ramp up the CFL used by the adaptive time step wizard during spin up.
    cfl(t) = min(0.01t, 0.1)

    function print_progress(simulation)
        model = simulation.model
        clock = model.clock

        wizard.cfl = cfl(model.clock.time)

        progress = 100 * (clock.time / end_time)

        umax = maximum(abs, model.velocities.u.data.parent)
        vmax = maximum(abs, model.velocities.v.data.parent)
        wmax = maximum(abs, model.velocities.w.data.parent)
        CFL = wizard.Δt / cell_advection_timescale(model)

        Δ = min(model.grid.Δx, model.grid.Δy, model.grid.Δz)
        νmax = maximum(model.diffusivities.νₑ.data.parent)
        κmax = maximum(model.diffusivities.κₑ.T.data.parent)
        νCFL = wizard.Δt / (Δ^2 / νmax)
        κCFL = wizard.Δt / (Δ^2 / κmax)

        @printf("[%06.2f%%] i: %d, t: %.2e, umax: (%.2e, %.2e, %.2e), CFL: %.2e, νκmax: (%.2e, %.2e), νκCFL: (%.2e, %.2e), next Δt: %.2e, wall time: %s\n",
                progress, model.clock.iteration, model.clock.time, umax, vmax, wmax,
                CFL, νmax, κmax, νCFL, κCFL, wizard.Δt, prettytime(simulation.run_time))
    end

    simulation = Simulation(model, Δt=wizard, stop_time=end_time,
                            progress=print_progress, iteration_interval=Ni)
    push!(simulation.output_writers, field_writer, profile_writer, statistics_writer)
    run!(simulation)

    return simulation
end
