using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization, VerticallyImplicitTimeDiscretization, z_viscosity
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary

function relative_error(u_num, u, time)
    u_ans = Field(location(u_num), architecture(u_num), u_num.grid, nothing)
    set!(u_ans, (x, y, z) -> u(x, y, z, time))
    return mean((interior(u_num) .- interior(u_ans)).^2 ) / mean(interior(u_ans).^2)
end

function test_diffusion_simple(fieldname, timestepper, time_discretization)

    model = IncompressibleModel(timestepper = timestepper,
                                       grid = RegularRectilinearGrid(size=(1, 1, 16), extent=(1, 1, 1)),
                                    closure = IsotropicDiffusivity(ν=1, κ=1, time_discretization=time_discretization),
                                   coriolis = nothing,
                                    tracers = :c,
                                   buoyancy = nothing)

    field = get_model_field(fieldname, model)

    value = π
    interior(field) .= value

    for n in 1:10
        ab2_or_rk3_time_step!(model, 1, n)
    end

    field_data = interior(field)

    return !any(@. !isapprox(value, field_data))
end

function test_isotropic_diffusion_budget(fieldname, model)
    set!(model; u=0, v=0, w=0, c=0)
    set!(model; Dict(fieldname => (x, y, z) -> rand())...)

    field = get_model_field(fieldname, model)

    ν = z_viscosity(model.closure, nothing) # for generalizing to isotropic AnisotropicDiffusivity

    return test_diffusion_budget(fieldname, field, model, ν, model.grid.Δz)
end

function test_biharmonic_diffusion_budget(fieldname, model)
    set!(model; u=0, v=0, w=0, c=0)
    set!(model; Dict(fieldname => (x, y, z) -> rand())...)

    field = get_model_field(fieldname, model)

    return test_diffusion_budget(fieldname, field, model, model.closure.νz, model.grid.Δz, 4)
end

function test_diffusion_budget(fieldname, field, model, κ, Δ, order=2)
    init_mean = mean(interior(field))

    for n in 1:10
        # Very small time-steps required to bring error under machine precision
        ab2_or_rk3_time_step!(model, 1e-4 * Δ^order / κ, n)
    end

    final_mean = mean(interior(field))

    @info @sprintf("    Initial <%s>: %.16f, final <%s>: %.16f, final - initial: %.4e",
                   fieldname, init_mean, fieldname, final_mean, final_mean - init_mean)

    return isapprox(init_mean, final_mean)
end

function test_diffusion_cosine(fieldname, timestepper, grid, time_discretization)
    κ, m = 1, 2 # diffusivity and cosine wavenumber

    model = IncompressibleModel(timestepper = timestepper,
                                       grid = grid,
                                    closure = IsotropicDiffusivity(ν=κ, κ=κ, time_discretization=time_discretization),
                                   buoyancy = nothing)

    field = get_model_field(fieldname, model)

    zC = znodes(Center, grid, reshape=true)
    interior(field) .= cos.(m * zC)

    diffusing_cosine(κ, m, z, t) = exp(-κ * m^2 * t) * cos(m * z)

    # Step forward with small time-step relative to diff. time-scale
    Δt = 1e-6 * grid.Lz^2 / κ
    for n in 1:10
        ab2_or_rk3_time_step!(model, Δt, n)
    end

     numerical = interior(field)
    analytical = diffusing_cosine.(κ, m, zC, model.clock.time)

    return !any(@. !isapprox(numerical, analytical, atol=1e-6, rtol=1e-6))
end

function internal_wave_test(timestepper; N=128, Nt=10, background_stratification=false)
    # Internal wave parameters
     ν = κ = 1e-9
     L = 2π
    z₀ = -L/3
     δ = L/20
    a₀ = 1e-3
     m = 16
     k = 1
     f = 0.2
     ℕ = 1.0
     σ = sqrt( (ℕ^2*k^2 + f^2*m^2) / (k^2 + m^2) )

    # Numerical parameters
     N = 128
    Δt = 0.01 * 1/σ

    cᵍ = m * σ / (k^2 + m^2) * (f^2/σ^2 - 1)
     U = a₀ * k * σ   / (σ^2 - f^2)
     V = a₀ * k * f   / (σ^2 - f^2)
     W = a₀ * m * σ   / (σ^2 - ℕ^2)
     B = a₀ * m * ℕ^2 / (σ^2 - ℕ^2)

    a(x, y, z, t) = exp( -(z - cᵍ*t - z₀)^2 / (2*δ)^2 )

    u(x, y, z, t) = a(x, y, z, t) * U * cos(k*x + m*z - σ*t)
    v(x, y, z, t) = a(x, y, z, t) * V * sin(k*x + m*z - σ*t)
    w(x, y, z, t) = a(x, y, z, t) * W * cos(k*x + m*z - σ*t)

    b(x, y, z, t) = ℕ^2 * z + a(x, y, z, t) * B * sin(k*x + m*z - σ*t)
    background_fields = NamedTuple()

    if background_stratification # Move stratification to a background field
        b(x, y, z, t) = a(x, y, z, t) * B * sin(k*x + m*z - σ*t)
        background_b(x, y, z, t) = ℕ^2 * z
        background_fields = (b=background_b,)
    end

    u₀(x, y, z) = u(x, y, z, 0)
    v₀(x, y, z) = v(x, y, z, 0)
    w₀(x, y, z) = w(x, y, z, 0)
    b₀(x, y, z) = b(x, y, z, 0)

    model = IncompressibleModel(timestepper = timestepper,
                                       grid = RegularRectilinearGrid(size=(N, 1, N), extent=(L, L, L)),
                                    closure = IsotropicDiffusivity(ν=ν, κ=κ),
                                   buoyancy = BuoyancyTracer(),
                          background_fields = background_fields,
                                    tracers = :b,
                                   coriolis = FPlane(f=f))

    set!(model, u=u₀, v=v₀, w=w₀, b=b₀)

    for n in 1:Nt
        ab2_or_rk3_time_step!(model, Δt, n)
    end

    # Tolerance was found by trial and error...
    return relative_error(model.velocities.u, u, model.clock.time) < 1e-4
end

function passive_tracer_advection_test(timestepper; N=128, κ=1e-12, Nt=100, background_velocity_field=false)
    L, U, V = 1.0, 0.5, 0.8
    δ, x₀, y₀ = L/15, L/2, L/2

    Δt = 0.05 * L/N / sqrt(U^2 + V^2)

    T(x, y, z, t) = exp( -((x - U*t - x₀)^2 + (y - V*t - y₀)^2) / (2*δ^2) )
    u₀(x, y, z) = U
    v₀(x, y, z) = V
    T₀(x, y, z) = T(x, y, z, 0)
    background_fields = Dict()

    if background_velocity_field
        background_fields[:u] = (x, y, z, t) -> U
        background_fields[:v] = (x, y, z, t) -> V
        u₀ = 0
        v₀ = 0
    end

    background_fields = NamedTuple{Tuple(keys(background_fields))}(values(background_fields))

    grid = RegularRectilinearGrid(size=(N, N, 2), extent=(L, L, L))
    closure = IsotropicDiffusivity(ν=κ, κ=κ)
    model = IncompressibleModel(timestepper=timestepper, grid=grid, closure=closure,
                                background_fields=background_fields)

    set!(model, u=u₀, v=v₀, T=T₀)

    for n in 1:Nt
        ab2_or_rk3_time_step!(model, Δt, n)
    end

    # Error tolerance is a bit arbitrary
    return relative_error(model.tracers.T, T, model.clock.time) < 1e-4
end

"""
Taylor-Green vortex test
See: https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex#Taylor%E2%80%93Green_vortex_solution
     and p. 310 of "Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Application"
     by Hesthaven & Warburton.
"""
function taylor_green_vortex_test(arch, timestepper, time_discretization; FT=Float64, N=64, Nt=10)
    Nx, Ny, Nz = N, N, 2
    Lx, Ly, Lz = 1, 1, 1
    ν = 1

    # Choose a very small time step as we are diffusion-limited in this test: Δt ≤ Δx² / 2ν
    Δx = Lx / Nx
    Δt = (1/10π) * Δx^2 / ν

    # Taylor-Green vortex analytic solution.
    @inline u(x, y, z, t) = -sin(2π*y) * exp(-4π^2 * ν * t)
    @inline v(x, y, z, t) =  sin(2π*x) * exp(-4π^2 * ν * t)

    model = IncompressibleModel(
        architecture = arch,
         timestepper = timestepper,
                grid = RegularRectilinearGrid(FT, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)),
             closure = IsotropicDiffusivity(FT, ν=1, time_discretization=time_discretization),
             tracers = nothing,
            buoyancy = nothing)

    u₀(x, y, z) = u(x, y, z, 0)
    v₀(x, y, z) = v(x, y, z, 0)
    set!(model, u=u₀, v=v₀)

    for n in 1:Nt
        ab2_or_rk3_time_step!(model, Δt, n)
    end

    xF, yC, zC = nodes(model.velocities.u, reshape=true)
    xC, yF, zC = nodes(model.velocities.v, reshape=true)

    t = model.clock.time
    i = model.clock.iteration

    # Calculate relative error between model and analytic solutions for u and v.
    u_rel_err = abs.((interior(model.velocities.u) .- u.(xF, yC, zC, t)) ./ u.(xF, yC, zC, t))
    u_rel_err_avg = mean(u_rel_err)
    u_rel_err_max = maximum(u_rel_err)

    v_rel_err = abs.((interior(model.velocities.v) .- v.(xC, yF, zC, t)) ./ v.(xC, yF, zC, t))
    v_rel_err_avg = mean(v_rel_err)
    v_rel_err_max = maximum(v_rel_err)

    @info "Taylor-Green vortex test [$arch, $FT, Nx=Ny=$N, Nt=$Nt]: " *
          @sprintf("Δu: (avg=%6.3g, max=%6.3g), Δv: (avg=%6.3g, max=%6.3g)",
                   u_rel_err_avg, u_rel_err_max, v_rel_err_avg, v_rel_err_max)

    return u_rel_err_max < 5e-6 && v_rel_err_max < 5e-6
end

function stratified_fluid_remains_at_rest_with_tilted_gravity_buoyancy_tracer(arch, FT; N=32, L=2000, θ=60, N²=1e-5)
    topo = (Periodic, Bounded, Bounded)
    grid = RegularRectilinearGrid(FT, topology=topo, size=(1, N, N), extent=(L, L, L))

    g̃ = (0, sind(θ), cosd(θ))
    buoyancy = Buoyancy(model=BuoyancyTracer(), vertical_unit_vector=g̃)

    y_bc = GradientBoundaryCondition(N² * g̃[2])
    z_bc = GradientBoundaryCondition(N² * g̃[3])
    b_bcs = FieldBoundaryConditions(bottom=z_bc, top=z_bc, south=y_bc, north=y_bc)

    model = IncompressibleModel(
               architecture = arch,
                       grid = grid,
                   buoyancy = buoyancy,
                    tracers = :b,
                    closure = nothing,
        boundary_conditions = (b=b_bcs,)
    )

    b₀(x, y, z) = N² * (x*g̃[1] + y*g̃[2] + z*g̃[3])
    set!(model, b=b₀)

    simulation = Simulation(model, Δt=10minutes, stop_time=1hour)
    run!(simulation)

    ∂y_b = ComputedField(∂y(model.tracers.b))
    ∂z_b = ComputedField(∂z(model.tracers.b))

    compute!(∂y_b)
    compute!(∂z_b)

    mean_∂y_b = mean(∂y_b)
    mean_∂z_b = mean(∂z_b)

    Δ_y = N² * g̃[2] - mean_∂y_b
    Δ_z = N² * g̃[3] - mean_∂z_b

    @info "N² * g̃[2] = $(N² * g̃[2]), mean(∂y_b) = $(mean_∂y_b), Δ = $Δ_y at t = $(prettytime(model.clock.time)) with θ=$(θ)°"
    @info "N² * g̃[3] = $(N² * g̃[3]), mean(∂z_b) = $(mean_∂z_b), Δ = $Δ_z at t = $(prettytime(model.clock.time)) with θ=$(θ)°"

    @test N² * g̃[2] ≈ mean(∂y_b)
    @test N² * g̃[3] ≈ mean(∂z_b)

    CUDA.@allowscalar begin
        @test all(N² * g̃[2] .≈ interior(∂y_b))
        @test all(N² * g̃[3] .≈ interior(∂z_b))
    end

    return nothing
end

function stratified_fluid_remains_at_rest_with_tilted_gravity_temperature_tracer(arch, FT; N=32, L=2000, θ=60, N²=1e-5)
    topo = (Periodic, Bounded, Bounded)
    grid = RegularRectilinearGrid(FT, topology=topo, size=(1, N, N), extent=(L, L, L))

    g̃ = (0, sind(θ), cosd(θ))
    buoyancy = Buoyancy(model=SeawaterBuoyancy(), vertical_unit_vector=g̃)

    α  = buoyancy.model.equation_of_state.α
    g₀ = buoyancy.model.gravitational_acceleration
    ∂T∂z = N² / (g₀ * α)

    y_bc = GradientBoundaryCondition(∂T∂z * g̃[2])
    z_bc = GradientBoundaryCondition(∂T∂z * g̃[3])
    T_bcs = FieldBoundaryConditions(bottom=z_bc, top=z_bc, south=y_bc, north=y_bc)

    model = IncompressibleModel(
               architecture = arch,
                       grid = grid,
                   buoyancy = buoyancy,
                    tracers = (:T, :S),
                    closure = nothing,
        boundary_conditions = (T=T_bcs,)
    )

    T₀(x, y, z) = ∂T∂z * (x*g̃[1] + y*g̃[2] + z*g̃[3])
    set!(model, T=T₀)

    simulation = Simulation(model, Δt=10minute, stop_time=1hour)
    run!(simulation)

    ∂y_T = ComputedField(∂y(model.tracers.T))
    ∂z_T = ComputedField(∂z(model.tracers.T))

    compute!(∂y_T)
    compute!(∂z_T)

    mean_∂y_T = mean(∂y_T)
    mean_∂z_T = mean(∂z_T)

    Δ_y = ∂T∂z * g̃[2] - mean_∂y_T
    Δ_z = ∂T∂z * g̃[3] - mean_∂z_T

    @info "∂T∂z * g̃[2] = $(∂T∂z * g̃[2]), mean(∂y_T) = $(mean_∂y_T), Δ = $Δ_y at t = $(prettytime(model.clock.time)) with θ=$(θ)°"
    @info "∂T∂z * g̃[3] = $(∂T∂z * g̃[3]), mean(∂z_T) = $(mean_∂z_T), Δ = $Δ_z at t = $(prettytime(model.clock.time)) with θ=$(θ)°"

    @test ∂T∂z * g̃[2] ≈ mean(∂y_T)
    @test ∂T∂z * g̃[3] ≈ mean(∂z_T)

    CUDA.@allowscalar begin
        @test all(∂T∂z * g̃[2] .≈ interior(∂y_T))
        @test all(∂T∂z * g̃[3] .≈ interior(∂z_T))
    end

    return nothing
end

timesteppers = (:QuasiAdamsBashforth2, :RungeKutta3)

@testset "Dynamics" begin
    @info "Testing dynamics..."

    @testset "Simple diffusion" begin
        @info "  Testing simple diffusion..."
        for fieldname in (:u, :v, :c), timestepper in timesteppers
            for time_discretization in (ExplicitTimeDiscretization(), VerticallyImplicitTimeDiscretization())
                @test test_diffusion_simple(fieldname, timestepper, time_discretization)
            end
        end
    end

    @testset "Budgets in isotropic diffusion" begin
        @info "  Testing model budgets with isotropic diffusion..."
        for timestepper in timesteppers
            for topology in ((Periodic, Periodic, Periodic),
                             (Periodic, Periodic, Bounded),
                             (Periodic, Bounded, Bounded),
                             (Bounded, Bounded, Bounded))

                if topology !== (Periodic, Periodic, Periodic) # can't use implicit time-stepping in vertically-periodic domains right now
                    time_discretizations = (ExplicitTimeDiscretization(), VerticallyImplicitTimeDiscretization())
                else
                    time_discretizations = (ExplicitTimeDiscretization(),)
                end

                for time_discretization in time_discretizations

                    for closure in (IsotropicDiffusivity(ν=1, κ=1, time_discretization=time_discretization),
                                    AnisotropicDiffusivity(νh=1, νz=1, κh=1, κz=1, time_discretization=time_discretization))

                        fieldnames = [:c]

                        topology[1] === Periodic && push!(fieldnames, :u)
                        topology[2] === Periodic && push!(fieldnames, :v)
                        topology[3] === Periodic && push!(fieldnames, :w)

                        grid = RegularRectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=topology)

                        model = IncompressibleModel(timestepper = timestepper,
                                                           grid = grid,
                                                        closure = closure,
                                                        tracers = :c,
                                                       coriolis = nothing,
                                                       buoyancy = nothing)

                        td = typeof(time_discretization).name.wrapper
                        closurename = typeof(closure).name.wrapper

                        for fieldname in fieldnames
                            @info "    [$timestepper, $td, $closurename] Testing $fieldname budget in a $topology domain with isotropic diffusion..."
                            @test test_isotropic_diffusion_budget(fieldname, model)
                        end
                    end
                end
            end
        end
    end

    @testset "Budgets in biharmonic diffusion" begin
        @info "  Testing model budgets with biharmonic diffusion..."
        for timestepper in timesteppers
            for topology in ((Periodic, Periodic, Periodic),
                             (Periodic, Periodic, Bounded),
                             (Periodic, Bounded, Bounded),
                             (Bounded, Bounded, Bounded))

                fieldnames = [:c]

                topology[1] === Periodic && push!(fieldnames, :u)
                topology[2] === Periodic && push!(fieldnames, :v)
                topology[3] === Periodic && push!(fieldnames, :w)

                grid = RegularRectilinearGrid(size=(2, 2, 2), extent=(1, 1, 1), topology=topology)

                model = IncompressibleModel(timestepper = timestepper,
                                                   grid = grid,
                                                closure = AnisotropicBiharmonicDiffusivity(νh=1, νz=1, κh=1, κz=1),
                                               coriolis = nothing,
                                                tracers = :c,
                                               buoyancy = nothing)

                for fieldname in fieldnames
                    @info "    [$timestepper] Testing $fieldname budget in a $topology domain with biharmonic diffusion..."
                    @test test_biharmonic_diffusion_budget(fieldname, model)
                end
            end
        end
    end

    @testset "Diffusion cosine" begin
        for timestepper in (:QuasiAdamsBashforth2,) #timesteppers
            for fieldname in (:u, :v, :T, :S)
                for time_discretization in (ExplicitTimeDiscretization(), VerticallyImplicitTimeDiscretization())
                    Nz, Lz = 128, π/2
                    grid = RegularRectilinearGrid(size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=(0, Lz))

                    @info "  Testing diffusion cosine [$fieldname, $timestepper, $time_discretization]..."
                    @test test_diffusion_cosine(fieldname, timestepper, grid, time_discretization)

                    @info "  Testing diffusion cosine on ImmersedBoundaryGrid [$fieldname, $timestepper, $time_discretization]..."
                    solid(x, y, z) = false
                    immersed_grid = ImmersedBoundaryGrid(grid, GridFittedBoundary(solid))
                    @test test_diffusion_cosine(fieldname, timestepper, immersed_grid, time_discretization)
                end
            end
        end
    end

    @testset "Passive tracer advection" begin
        for timestepper in (:QuasiAdamsBashforth2,) #timesteppers
            @info "  Testing passive tracer advection [$timestepper]..."
            @test passive_tracer_advection_test(timestepper)
        end
    end

    @testset "Internal wave" begin
        for timestepper in (:QuasiAdamsBashforth2,) #timesteppers
            @info "  Testing internal wave [$timestepper]..."
            @test internal_wave_test(timestepper)
        end
    end

    @testset "Taylor-Green vortex" begin
        for timestepper in (:QuasiAdamsBashforth2,) #timesteppers
            for time_discretization in (ExplicitTimeDiscretization(), VerticallyImplicitTimeDiscretization())
                td = typeof(time_discretization).name.wrapper
                @info "  Testing Taylor-Green vortex [$timestepper, $td]..."
                @test taylor_green_vortex_test(CPU(), timestepper, time_discretization)
            end
        end
    end

    @testset "Background fields" begin
        for timestepper in (:QuasiAdamsBashforth2,) #timesteppers
            @info "  Testing dynamics with background fields [$timestepper]..."
            @test_skip passive_tracer_advection_test(timestepper, background_velocity_field=true)
            @test internal_wave_test(timestepper, background_stratification=true)
        end
    end

    @testset "Tilted gravity" begin
        for arch in archs
            @info "  Testing tilted gravity [$(typeof(arch))]..."
            for θ in (0, 1, -30, 60, 90, -180)
                stratified_fluid_remains_at_rest_with_tilted_gravity_buoyancy_tracer(arch, Float64, θ=θ)
                stratified_fluid_remains_at_rest_with_tilted_gravity_temperature_tracer(arch, Float64, θ=θ)
            end
        end
    end
end
