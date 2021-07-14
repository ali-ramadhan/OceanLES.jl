using Oceananigans.BoundaryConditions: ContinuousBoundaryFunction

function test_boundary_condition(arch, FT, topo, side, field_name, boundary_condition)
    grid = RegularRectilinearGrid(FT, size=(1, 1, 1), extent=(1, π, 42), topology=topo)

    boundary_condition_kwarg = (; side => boundary_condition)
    field_boundary_conditions = FieldBoundaryConditions(; boundary_condition_kwarg...)
    bcs = (; field_name => field_boundary_conditions)
    model = IncompressibleModel(grid=grid, architecture=arch, boundary_conditions=bcs)

    success = try
        time_step!(model, 1e-16, euler=true)
        true
    catch err
        @warn "test_boundary_condition errored with " * sprint(showerror, err)
        false
    end

    return success
end

function test_incompressible_flux_budget(arch, name, side, topo)

    FT = Float64
    Lx = 0.3
    Ly = 0.4
    Lz = 0.5

    grid = RegularRectilinearGrid(FT,
                                  size = (1, 1, 1),
                                  x = (0, Lx),
                                  y = (0, Ly),
                                  z = (0, Lz),
                                  topology = topo)

    flux = FT(π)
    direction = side ∈ (:west, :south, :bottom) ? 1 : -1
    bc_kwarg = Dict(side => BoundaryCondition(Flux, flux * direction))

    field_bcs = FieldBoundaryConditions(; bc_kwarg...)

    model_bcs = NamedTuple{tuple(name)}(tuple(field_bcs))

    model = IncompressibleModel(grid=grid, buoyancy=nothing, boundary_conditions=model_bcs,
                                closure=nothing, architecture=arch, tracers=:c)
                                
    is_velocity_field = name ∈ (:u, :v, :w)
    field = is_velocity_field ? getproperty(model.velocities, name) : getproperty(model.tracers, name)
    set!(field, 0)

    simulation = Simulation(model, Δt = 1.0, stop_iteration = 1)
    run!(simulation)

    mean_ϕ = CUDA.@allowscalar field[1, 1, 1]

    L = side ∈ (:west, :east) ? Lx :
        side ∈ (:south, :north) ? Ly : Lz

    # budget: L * ∂<ϕ>/∂t = -Δflux = -flux / L (left) + flux / L (right)
    # therefore <ϕ> = flux * t / L
    #
    # Note \approx, because velocity budgets are off by machine precision (due to pressure solve?)
    return mean_ϕ ≈ flux * model.clock.time / L
end

function fluxes_with_diffusivity_boundary_conditions_are_correct(arch, FT)
    Lz = 1
    κ₀ = FT(exp(-3))
    bz = FT(π)
    flux = - κ₀ * bz

    grid = RegularRectilinearGrid(FT, size=(16, 16, 16), extent=(1, 1, Lz))

    buoyancy_bcs = FieldBoundaryConditions(bottom=GradientBoundaryCondition(bz))
    κₑ_bcs = FieldBoundaryConditions(grid, (Center, Center, Center), bottom=ValueBoundaryCondition(κ₀))
    model_bcs = (b=buoyancy_bcs, κₑ=(b=κₑ_bcs,))

    model = IncompressibleModel(
        grid=grid, architecture=arch, tracers=:b, buoyancy=BuoyancyTracer(),
        closure=AnisotropicMinimumDissipation(), boundary_conditions=model_bcs
    )

    b₀(x, y, z) = z * bz
    set!(model, b=b₀)

    b = model.tracers.b
    mean_b₀ = mean(interior(b))

    τκ = Lz^2 / κ₀  # Diffusion time-scale
    Δt = 1e-6 * τκ  # Time step much less than diffusion time-scale
    Nt = 10         # Number of time steps

    for n in 1:Nt
        time_step!(model, Δt, euler= n==1)
    end

    # budget: Lz*∂<ϕ>/∂t = -Δflux = -top_flux/Lz (left) + bottom_flux/Lz (right)
    # therefore <ϕ> = bottom_flux * t / Lz
    #
    # Use an atol of 1e-6 so test passes with Float32 as there's a big cancellation
    # error due to buoyancy order of magnitude.
    #
    # Float32:
    # mean_b₀ = -1.5707965f0
    # mean(interior(b)) = -1.5708286f0
    # mean(interior(b)) - mean_b₀ = -3.20673f-5
    # (flux * model.clock.time) / Lz = -3.141593f-5
    #
    # Float64
    # mean_b₀ = -1.5707963267949192
    # mean(interior(b)) = -1.57082774272148
    # mean(interior(b)) - mean_b₀ = -3.141592656086267e-5
    # (flux * model.clock.time) / Lz = -3.141592653589793e-5
    
    return isapprox(mean(interior(b)) - mean_b₀, flux * model.clock.time / Lz, atol=1e-6)
end

test_boundary_conditions(C, FT, ArrayType) = (integer_bc(C, FT, ArrayType),
                                              float_bc(C, FT, ArrayType),
                                              irrational_bc(C, FT, ArrayType),
                                              array_bc(C, FT, ArrayType),
                                              simple_function_bc(C, FT, ArrayType),
                                              parameterized_function_bc(C, FT, ArrayType),
                                              field_dependent_function_bc(C, FT, ArrayType),
                                              parameterized_field_dependent_function_bc(C, FT, ArrayType),
                                              discrete_function_bc(C, FT, ArrayType),
                                              parameterized_discrete_function_bc(C, FT, ArrayType)
                                             )

@testset "Boundary condition integration tests" begin
    @info "Testing boundary condition integration into IncompressibleModel..."

    @testset "Boundary condition regularization" begin
        @info "  Testing boundary condition regularization in IncompressibleModel constructor..."

        FT = Float64
        arch = first(archs)

        # We use Periodic, Bounded, Bounded here because triply Bounded domains don't work on the GPU
        # yet.
        grid = RegularRectilinearGrid(FT, size=(1, 1, 1), extent=(1, π, 42), topology=(Periodic, Bounded, Bounded))

        u_boundary_conditions = FieldBoundaryConditions(bottom = simple_function_bc(Value),
                                                        top    = simple_function_bc(Value),
                                                        north  = simple_function_bc(Value),
                                                        south  = simple_function_bc(Value))

        v_boundary_conditions = FieldBoundaryConditions(bottom = simple_function_bc(Value),
                                                        top    = simple_function_bc(Value),
                                                        north  = simple_function_bc(Open),
                                                        south  = simple_function_bc(Open))


        w_boundary_conditions = FieldBoundaryConditions(bottom = simple_function_bc(Open),
                                                        top    = simple_function_bc(Open),
                                                        north  = simple_function_bc(Value),
                                                        south  = simple_function_bc(Value))

        T_boundary_conditions = FieldBoundaryConditions(bottom = simple_function_bc(Value),
                                                        top    = simple_function_bc(Value),
                                                        north  = simple_function_bc(Value),
                                                        south  = simple_function_bc(Value))

        boundary_conditions = (u=u_boundary_conditions,
                               v=v_boundary_conditions,
                               w=w_boundary_conditions,
                               T=T_boundary_conditions)

        model = IncompressibleModel(architecture = arch,
                                    grid = grid,
                                    boundary_conditions = boundary_conditions)

        @test location(model.velocities.u.boundary_conditions.bottom.condition) == (Face, Center, Nothing)
        @test location(model.velocities.u.boundary_conditions.top.condition)    == (Face, Center, Nothing)
        @test location(model.velocities.u.boundary_conditions.north.condition)  == (Face, Nothing, Center)
        @test location(model.velocities.u.boundary_conditions.south.condition)  == (Face, Nothing, Center)

        @test location(model.velocities.v.boundary_conditions.bottom.condition) == (Center, Face, Nothing)
        @test location(model.velocities.v.boundary_conditions.top.condition)    == (Center, Face, Nothing)
        @test location(model.velocities.v.boundary_conditions.north.condition)  == (Center, Nothing, Center)
        @test location(model.velocities.v.boundary_conditions.south.condition)  == (Center, Nothing, Center)

        @test location(model.velocities.w.boundary_conditions.bottom.condition) == (Center, Center, Nothing)
        @test location(model.velocities.w.boundary_conditions.top.condition)    == (Center, Center, Nothing)
        @test location(model.velocities.w.boundary_conditions.north.condition)  == (Center, Nothing, Face)
        @test location(model.velocities.w.boundary_conditions.south.condition)  == (Center, Nothing, Face)

        @test location(model.tracers.T.boundary_conditions.bottom.condition) == (Center, Center, Nothing)
        @test location(model.tracers.T.boundary_conditions.top.condition)    == (Center, Center, Nothing)
        @test location(model.tracers.T.boundary_conditions.north.condition)  == (Center, Nothing, Center)
        @test location(model.tracers.T.boundary_conditions.south.condition)  == (Center, Nothing, Center)
    end

    @testset "Boudnary condition time-stepping works" begin
        for arch in archs, FT in (Float64,) #float_types
            @info "  Testing that time-stepping with boundary conditions works [$(typeof(arch)), $FT]..."

            topo = arch isa CPU ? (Bounded, Bounded, Bounded) : (Periodic, Bounded, Bounded)

            for C in (Gradient, Flux, Value), boundary_condition in test_boundary_conditions(C, FT, array_type(arch))
                arch isa CPU && @test test_boundary_condition(arch, FT, topo, :east, :T, boundary_condition)

                @test test_boundary_condition(arch, FT, topo, :south, :T, boundary_condition)
                @test test_boundary_condition(arch, FT, topo, :top, :T, boundary_condition)
            end

            for boundary_condition in test_boundary_conditions(Open, FT, array_type(arch))
                 arch isa CPU && @test test_boundary_condition(arch, FT, topo, :east, :u, boundary_condition)

                @test test_boundary_condition(arch, FT, topo, :south, :v, boundary_condition)
                @test test_boundary_condition(arch, FT, topo, :top, :w, boundary_condition)
            end
        end
    end

    @testset "Budgets with Flux boundary conditions" begin
        for arch in archs
            @info "  Testing budgets with Flux boundary conditions [$(typeof(arch))]..."

            topo = (Periodic, Bounded, Bounded)
            for name in (:u, :c), side in (:north, :south, :top, :bottom)
                @info "    Testing budgets with Flux boundary conditions [$(typeof(arch)), $topo, $name, $side]..."
                @test test_incompressible_flux_budget(arch, name, side, topo)
            end

            topo = (Bounded, Periodic, Bounded)
            for name in (:v, :c), side in (:east, :west, :top, :bottom)
                @info "    Testing budgets with Flux boundary conditions [$(typeof(arch)), $topo, $name, $side]..."
                @test test_incompressible_flux_budget(arch, name, side, topo)
            end

            topo = (Bounded, Bounded, Periodic)
            for name in (:w, :c), side in (:east, :west, :north, :south)
                @info "    Testing budgets with Flux boundary conditions [$(typeof(arch)), $topo, $name, $side]..."
                @test test_incompressible_flux_budget(arch, name, side, topo)
            end
        end
    end

    @testset "Custom diffusivity boundary conditions" begin
        for arch in archs, FT in (Float64,) #float_types
            @info "  Testing flux budgets with diffusivity boundary conditions [$(typeof(arch)), $FT]..."
            @test fluxes_with_diffusivity_boundary_conditions_are_correct(arch, FT)
        end
    end
end
