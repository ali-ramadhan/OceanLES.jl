using Oceananigans.Grids: xnode, znode
using Oceananigans.TimeSteppers: update_state!

function run_rayleigh_benard_regression_test(arch, grid_type)

    #####
    ##### Parameters
    #####
          α = 2                 # aspect ratio
          n = 1                 # resolution multiple
         Ra = 1e6               # Rayleigh number
    Nx = Ny = 8n * α            # horizontal resolution
    Lx = Ly = 1.0 * α           # horizontal extent
         Nz = 16n               # vertical resolution
         Lz = 1.0               # vertical extent
         Pr = 0.7               # Prandtl number
          a = 1e-1              # noise amplitude for initial condition
         Δb = 1.0               # buoyancy differential

    # Rayleigh and Prandtl determine transport coefficients
    ν = sqrt(Δb * Pr * Lz^3 / Ra)
    κ = ν / Pr

    #####
    ##### Model setup
    #####

    if grid_type == :regular
        grid = RegularRectilinearGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    elseif grid_type == :vertically_unstretched
        zF = range(-Lz, 0, length=Nz+1)
        grid = VerticallyStretchedRectilinearGrid(architecture=arch, size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z_faces=zF)
    end

    # Force salinity as a passive tracer (βS=0)
    c★(x, z) = exp(4z) * sin(2π/Lx * x)
    Fc(i, j, k, grid, clock, model_fields) = 1/10 * (c★(xnode(Center(), i, grid), znode(Center(), k, grid)) - model_fields.c[i, j, k])

    bbcs = FieldBoundaryConditions(top = BoundaryCondition(Value, 0.0),
                                   bottom = BoundaryCondition(Value, Δb))

    model = IncompressibleModel(
               architecture = arch,
                       grid = grid,
                    closure = IsotropicDiffusivity(ν=ν, κ=κ),
                    tracers = (:b, :c),
                   buoyancy = Buoyancy(model=BuoyancyTracer()),
        boundary_conditions = (b=bbcs,),
                    forcing = (c=Forcing(Fc, discrete_form=true),)
    )

    # Lz/Nz will work for both the :regular and :vertically_unstretched grids.
    Δt = 0.01 * min(model.grid.Δx, model.grid.Δy, Lz/Nz)^2 / ν

    # We will manually change the stop_iteration as needed.
    simulation = Simulation(model, Δt=Δt, stop_iteration=0)

    # The type of the underlying data, not the offset array.
    ArrayType = typeof(model.velocities.u.data.parent)

    spinup_steps = 1000
      test_steps = 100

    prefix = "rayleigh_benard"

    checkpointer = Checkpointer(model, schedule=IterationInterval(test_steps), prefix=prefix,
                                dir=joinpath(dirname(@__FILE__), "data"))

    #####
    ##### Initial condition and spinup steps for creating regression test data
    #####

    #=
    @warn "Generating new data for the Rayleigh-Benard regression test."

    ξ(z) = a * rand() * z * (Lz + z) # noise, damped at the walls
    b₀(x, y, z) = (ξ(z) - z) / Lz
    set!(model, b=b₀)

    simulation.stop_iteration = spinup_steps-test_steps
    run!(simulation)

    push!(simulation.output_writers, checkpointer)
    simulation.stop_iteration += 2test_steps
    run!(simulation)
    =#

    #####
    ##### Regression test
    #####

    # Load initial state
    initial_filename = joinpath(dirname(@__FILE__), "data", prefix * "_iteration$spinup_steps.jld2")

    solution₀, Gⁿ₀, G⁻₀ = get_fields_from_checkpoint(initial_filename)

    model.velocities.u.data.parent .= ArrayType(solution₀.u)
    model.velocities.v.data.parent .= ArrayType(solution₀.v)
    model.velocities.w.data.parent .= ArrayType(solution₀.w)
    model.tracers.b.data.parent    .= ArrayType(solution₀.b)
    model.tracers.c.data.parent    .= ArrayType(solution₀.c)

    model.timestepper.Gⁿ.u.data.parent .= ArrayType(Gⁿ₀.u)
    model.timestepper.Gⁿ.v.data.parent .= ArrayType(Gⁿ₀.v)
    model.timestepper.Gⁿ.w.data.parent .= ArrayType(Gⁿ₀.w)
    model.timestepper.Gⁿ.b.data.parent .= ArrayType(Gⁿ₀.b)
    model.timestepper.Gⁿ.c.data.parent .= ArrayType(Gⁿ₀.c)

    model.timestepper.G⁻.u.data.parent .= ArrayType(G⁻₀.u)
    model.timestepper.G⁻.v.data.parent .= ArrayType(G⁻₀.v)
    model.timestepper.G⁻.w.data.parent .= ArrayType(G⁻₀.w)
    model.timestepper.G⁻.b.data.parent .= ArrayType(G⁻₀.b)
    model.timestepper.G⁻.c.data.parent .= ArrayType(G⁻₀.c)

    model.clock.iteration = spinup_steps
    model.clock.time = spinup_steps * Δt
    length(simulation.output_writers) > 0 && pop!(simulation.output_writers)

    # Step the model forward and perform the regression test
    update_state!(model)

    for n in 1:test_steps
        time_step!(model, Δt, euler=false)
    end

    final_filename = joinpath(@__DIR__, "data", prefix * "_iteration$(spinup_steps+test_steps).jld2")

    solution₁, Gⁿ₁, G⁻₁ = get_fields_from_checkpoint(final_filename)

    test_fields =  CUDA.@allowscalar (u = Array(interior(model.velocities.u)),
                                      v = Array(interior(model.velocities.v)),
                                      w = Array(interior(model.velocities.w)[:, :, 1:Nz]),
                                      b = Array(interior(model.tracers.b)),
                                      c = Array(interior(model.tracers.c)))

    correct_fields = (u = Array(interior(solution₁.u, model.grid)),
                      v = Array(interior(solution₁.v, model.grid)),
                      w = Array(interior(solution₁.w, model.grid)),
                      b = Array(interior(solution₁.b, model.grid)),
                      c = Array(interior(solution₁.c, model.grid)))

    summarize_regression_test(test_fields, correct_fields)

    CUDA.allowscalar(true)
    @test all(test_fields.u .≈ correct_fields.u)
    @test all(test_fields.v .≈ correct_fields.v)
    @test all(test_fields.w .≈ correct_fields.w)
    @test all(test_fields.b .≈ correct_fields.b)
    @test all(test_fields.c .≈ correct_fields.c)
    CUDA.allowscalar(false)

    return nothing
end
