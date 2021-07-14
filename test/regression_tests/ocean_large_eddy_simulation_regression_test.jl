using Oceananigans.TurbulenceClosures: AnisotropicMinimumDissipation
using Oceananigans.TimeSteppers: update_state!

function run_ocean_large_eddy_simulation_regression_test(arch, grid_type, closure)
    name = "ocean_large_eddy_simulation_" * string(typeof(closure).name.wrapper)

    spinup_steps = 10000
      test_steps = 10
              Δt = 2.0

    # Parameters
      Qᵀ = 5e-5     # Temperature flux at surface
      Qᵘ = -2e-5    # Velocity flux at surface
    ∂T∂z = 0.005    # Initial vertical temperature gradient

    # Grid
    N = L = 16
    if grid_type == :regular
        grid = RegularRectilinearGrid(size=(N, N, N), extent=(L, L, L))
    elseif grid_type == :vertically_unstretched
        zF = range(-L, 0, length=N+1)
        grid = VerticallyStretchedRectilinearGrid(architecture=arch, size=(N, N, N), x=(0, L), y=(0, L), z_faces=zF)
    end

    # Boundary conditions
    u_bcs = FieldBoundaryConditions(top = BoundaryCondition(Flux, Qᵘ))
    T_bcs = FieldBoundaryConditions(top = BoundaryCondition(Flux, Qᵀ), bottom = BoundaryCondition(Gradient, ∂T∂z))
    S_bcs = FieldBoundaryConditions(top = BoundaryCondition(Flux, 5e-8))

    # Model instantiation
    model = IncompressibleModel(
             architecture = arch,
                     grid = grid,
                 coriolis = FPlane(f=1e-4),
                 buoyancy = Buoyancy(model=SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4, β=8e-4))),
                  closure = closure,
      boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs)
    )

    # We will manually change the stop_iteration as needed.
    simulation = Simulation(model, Δt=Δt, stop_iteration=0)

    # The type of the underlying data, not the offset array.
    ArrayType = typeof(model.velocities.u.data.parent)

    ####
    #### Uncomment the block below to generate regression data.
    ####

    #=
    @warn "Generating new data for the ocean LES regression test."

    # Initialize model: random noise damped at top and bottom
    Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise
    T₀(x, y, z) = 20 + ∂T∂z * z + ∂T∂z * model.grid.Lz * 1e-2 * Ξ(z)
    u₀(x, y, z) = sqrt(abs(Qᵘ)) * 1e-3 * Ξ(z)
    set!(model, u=u₀, w=u₀, T=T₀, S=35)

    simulation.stop_iteration = spinup_steps-test_steps
    run!(simulation)

    checkpointer = Checkpointer(model, schedule = IterationInterval(test_steps), prefix = name,
                                dir = joinpath(dirname(@__FILE__), "data"))

    simulation.output_writers[:checkpointer] = checkpointer

    simulation.stop_iteration += 2test_steps
    run!(simulation)
    pop!(simulation.output_writers, :checkpointer)
    =#

    ####
    #### Regression test
    ####

    initial_filename = joinpath(dirname(@__FILE__), "data", name * "_iteration$spinup_steps.jld2")

    solution₀, Gⁿ₀, G⁻₀ = get_fields_from_checkpoint(initial_filename)

    Nz = grid.Nz

    model.velocities.u.data.parent .= ArrayType(solution₀.u)
    model.velocities.v.data.parent .= ArrayType(solution₀.v)
    model.velocities.w.data.parent .= ArrayType(solution₀.w)
    model.tracers.T.data.parent    .= ArrayType(solution₀.T)
    model.tracers.S.data.parent    .= ArrayType(solution₀.S)

    model.timestepper.Gⁿ.u.data.parent .= ArrayType(Gⁿ₀.u)
    model.timestepper.Gⁿ.v.data.parent .= ArrayType(Gⁿ₀.v)
    model.timestepper.Gⁿ.w.data.parent .= ArrayType(Gⁿ₀.w)
    model.timestepper.Gⁿ.T.data.parent .= ArrayType(Gⁿ₀.T)
    model.timestepper.Gⁿ.S.data.parent .= ArrayType(Gⁿ₀.S)

    model.timestepper.G⁻.u.data.parent .= ArrayType(G⁻₀.u)
    model.timestepper.G⁻.v.data.parent .= ArrayType(G⁻₀.v)
    model.timestepper.G⁻.w.data.parent .= ArrayType(G⁻₀.w)
    model.timestepper.G⁻.T.data.parent .= ArrayType(G⁻₀.T)
    model.timestepper.G⁻.S.data.parent .= ArrayType(G⁻₀.S)

    model.clock.time = spinup_steps * Δt
    model.clock.iteration = spinup_steps

    update_state!(model)

    for n in 1:test_steps
        time_step!(model, Δt, euler=false)
    end

    final_filename = joinpath(@__DIR__, "data", name * "_iteration$(spinup_steps+test_steps).jld2")

    solution₁, Gⁿ₁, G⁻₁ = get_fields_from_checkpoint(final_filename)

    test_fields = CUDA.@allowscalar (u = Array(interior(model.velocities.u)),
                                     v = Array(interior(model.velocities.v)),
                                     w = Array(interior(model.velocities.w)[:, :, 1:Nz]),
                                     T = Array(interior(model.tracers.T)),
                                     S = Array(interior(model.tracers.S)))

    correct_fields = (u = Array(interior(solution₁.u, model.grid)),
                      v = Array(interior(solution₁.v, model.grid)),
                      w = Array(interior(solution₁.w, model.grid)),
                      T = Array(interior(solution₁.T, model.grid)),
                      S = Array(interior(solution₁.S, model.grid)))

    summarize_regression_test(test_fields, correct_fields)

    @test all(test_fields.u .≈ correct_fields.u)
    @test all(test_fields.v .≈ correct_fields.v)
    @test all(test_fields.w .≈ correct_fields.w)
    @test all(test_fields.T .≈ correct_fields.T)
    @test all(test_fields.S .≈ correct_fields.S)

    return nothing
end
