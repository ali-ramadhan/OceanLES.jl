using Test
using Statistics
using JLD2

using Oceananigans
using Oceananigans.Architectures: array_type

using Oceananigans.Fields: location

function generate_some_interesting_simulation_data(Nx, Ny, Nz; architecture=CPU())
    grid = RegularRectilinearGrid(size=(Nx, Ny, Nz), extent=(64, 64, 32))

    T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(5e-5), bottom = GradientBoundaryCondition(0.01))
    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(-3e-4))

    @inline Qˢ(x, y, t, S, evaporation_rate) = - evaporation_rate * S
    evaporation_bc = FluxBoundaryCondition(Qˢ, field_dependencies=:S, parameters=3e-7)
    S_bcs = FieldBoundaryConditions(top=evaporation_bc)

    model = IncompressibleModel(
               architecture = architecture,
                       grid = grid,
        boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs)
    )

    dTdz = 0.01
    Tᵢ(x, y, z) = 20 + dTdz * z + 1e-6 * randn()
    uᵢ(x, y, z) = 1e-3 * randn()
    set!(model, u=uᵢ, w=uᵢ, T=Tᵢ, S=35)

    wizard = TimeStepWizard(cfl=1.0, Δt=10.0, max_change=1.1, max_Δt=1minute)
    simulation = Simulation(model, Δt=wizard, stop_time=2minutes)

    u, v, w = model.velocities

    computed_fields = (
        b = BuoyancyField(model),
        ζ = ComputedField(∂x(v) - ∂y(u)),
        ke = ComputedField(√(u^2 + v^2))
    )

    fields_to_output = merge(model.velocities, model.tracers, computed_fields)

    simulation.output_writers[:jld2_3d_with_halos] =
        JLD2OutputWriter(model, fields_to_output,
                  prefix = "test_3d_output_with_halos",
            field_slicer = FieldSlicer(with_halos=true),
                schedule = TimeInterval(30seconds),
                   force = true)

    profiles = NamedTuple{keys(fields_to_output)}(AveragedField(f, dims=(1, 2)) for f in fields_to_output)

    simulation.output_writers[:jld2_1d_with_halos] =
        JLD2OutputWriter(model, profiles,
                  prefix = "test_1d_output_with_halos",
            field_slicer = FieldSlicer(with_halos=true),
                schedule = TimeInterval(30seconds),
                   force = true)

    run!(simulation)

    return nothing
end

@testset "OutputReaders" begin
    @info "Testing output readers..."

    Nx, Ny, Nz = 16, 10, 5
    generate_some_interesting_simulation_data(Nx, Ny, Nz)
    Nt = 5

    filepath3d = "test_3d_output_with_halos.jld2"
    filepath1d = "test_1d_output_with_halos.jld2"

    for arch in archs
        @testset "FieldTimeSeries{InMemory} [$(typeof(arch))]" begin
            @info "  Testing FieldTimeSeries{InMemory} [$(typeof(arch))]..."

            ## 3D Fields

            u3 = FieldTimeSeries(filepath3d, "u", architecture=arch)
            v3 = FieldTimeSeries(filepath3d, "v", architecture=arch)
            w3 = FieldTimeSeries(filepath3d, "w", architecture=arch)
            T3 = FieldTimeSeries(filepath3d, "T", architecture=arch)
            b3 = FieldTimeSeries(filepath3d, "b", architecture=arch)
            ζ3 = FieldTimeSeries(filepath3d, "ζ", architecture=arch)

            @test location(u3) == (Face, Center, Center)
            @test location(v3) == (Center, Face, Center)
            @test location(w3) == (Center, Center, Face)
            @test location(T3) == (Center, Center, Center)
            @test location(b3) == (Center, Center, Center)
            @test location(ζ3) == (Face, Face, Center)

            @test size(u3) == (Nx, Ny, Nz,   Nt)
            @test size(v3) == (Nx, Ny, Nz,   Nt)
            @test size(w3) == (Nx, Ny, Nz+1, Nt)
            @test size(T3) == (Nx, Ny, Nz,   Nt)
            @test size(b3) == (Nx, Ny, Nz,   Nt)
            @test size(ζ3) == (Nx, Ny, Nz,   Nt)

            ArrayType = array_type(arch)
            for fts in (u3, v3, w3, T3, b3, ζ3)
                @test fts.data.parent isa ArrayType
            end

            if arch isa CPU
                @test u3[1, 2, 3, 4] isa Number
                @test u3[1] isa Field
                @test v3[2] isa Field
            end

            ## 1D AveragedFields

            u1 = FieldTimeSeries(filepath1d, "u", architecture=arch)
            v1 = FieldTimeSeries(filepath1d, "v", architecture=arch)
            w1 = FieldTimeSeries(filepath1d, "w", architecture=arch)
            T1 = FieldTimeSeries(filepath1d, "T", architecture=arch)
            b1 = FieldTimeSeries(filepath1d, "b", architecture=arch)
            ζ1 = FieldTimeSeries(filepath1d, "ζ", architecture=arch)

            @test location(u1) == (Nothing, Nothing, Center)
            @test location(v1) == (Nothing, Nothing, Center)
            @test location(w1) == (Nothing, Nothing, Face)
            @test location(T1) == (Nothing, Nothing, Center)
            @test location(b1) == (Nothing, Nothing, Center)
            @test location(ζ1) == (Nothing, Nothing, Center)

            @test size(u1) == (1, 1, Nz,   Nt)
            @test size(v1) == (1, 1, Nz,   Nt)
            @test size(w1) == (1, 1, Nz+1, Nt)
            @test size(T1) == (1, 1, Nz,   Nt)
            @test size(b1) == (1, 1, Nz,   Nt)
            @test size(ζ1) == (1, 1, Nz,   Nt)

            for fts in (u1, v1, w1, T1, b1, ζ1)
                @test fts.data.parent isa ArrayType
            end

            if arch isa CPU
                @test u1[1, 1, 3, 4] isa Number
                @test u1[1] isa Field
                @test v1[2] isa Field
            end
        end
    end

    for arch in archs
        @testset "FieldTimeSeries{OnDisk} [$(typeof(arch))]" begin
            @info "  Testing FieldTimeSeries{OnDisk} [$(typeof(arch))]..."

            ArrayType = array_type(arch)

            ζ = FieldTimeSeries(filepath3d, "ζ", backend=OnDisk(), architecture=arch)
            @test location(ζ) == (Face, Face, Center)
            @test size(ζ) == (Nx, Ny, Nz, Nt)
            @test ζ[1] isa Field
            @test ζ[2] isa Field
            @test ζ[1].data.parent isa ArrayType

            b = FieldTimeSeries(filepath1d, "b", backend=OnDisk(), architecture=arch)
            @test location(b) == (Nothing, Nothing, Center)
            @test size(b) == (1, 1, Nz, Nt)
            @test b[1] isa Field
            @test b[2] isa Field
        end
    end

    for arch in archs
        @testset "FieldTimeSeries{InMemory} reductions" begin
            @info "  Testing FieldTimeSeries{InMemory} reductions..."

            for name in ("u", "v", "w", "T", "b", "ζ"), fun in (sum, mean, maximum, minimum)
                f = FieldTimeSeries(filepath3d, name, architecture=CPU())

                ε = eps(maximum(abs, f.data.parent))

                val1 = fun(f)
                val2 = fun([fun(f[n]) for n in 1:Nt])

                @test val1 ≈ val2 atol=4ε
            end
        end
    end

    for Backend in [InMemory, OnDisk]
        @testset "FieldDataset{$Backend}" begin
            @info "  Testing FieldDataset{$Backend}..."

            ds = FieldDataset(filepath3d, backend=Backend())

            @test ds isa Dict
            @test length(keys(ds)) == 8
            @test ds["u"] isa FieldTimeSeries
            @test ds["v"][1] isa Field
            @test ds["T"][2] isa Field
        end
    end

    rm("test_3d_output_with_halos.jld2")
    rm("test_1d_output_with_halos.jld2")
end
