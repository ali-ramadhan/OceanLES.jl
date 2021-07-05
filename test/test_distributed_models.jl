using MPI

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Distributed: index2rank, east_halo, west_halo, north_halo, south_halo, top_halo, bottom_halo

# Right now just testing with 4 ranks!
comm = MPI.COMM_WORLD
mpi_ranks = MPI.Comm_size(comm)
@assert mpi_ranks == 4

#####
##### Multi architectures and rank connectivity
#####

function test_triply_periodic_rank_connectivity_with_411_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularRectilinearGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(4, 1, 1))

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    @test local_rank == index2rank(arch.local_index..., arch.ranks...)

    connectivity = arch.connectivity

    # No communication in y and z.
    @test isnothing(connectivity.south)
    @test isnothing(connectivity.north)
    @test isnothing(connectivity.top)
    @test isnothing(connectivity.bottom)

    # +---+---+---+---+
    # | 0 | 1 | 2 | 3 |
    # +---+---+---+---+

    if local_rank == 0
        @test connectivity.east == 1
        @test connectivity.west == 3
    elseif local_rank == 1
        @test connectivity.east == 2
        @test connectivity.west == 0
    elseif local_rank == 2
        @test connectivity.east == 3
        @test connectivity.west == 1
    elseif local_rank == 3
        @test connectivity.east == 0
        @test connectivity.west == 2
    end

    return nothing
end

function test_triply_periodic_rank_connectivity_with_141_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularRectilinearGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(1, 4, 1))

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    @test local_rank == index2rank(arch.local_index..., arch.ranks...)

    connectivity = arch.connectivity

    # No communication in x and z.
    @test isnothing(connectivity.east)
    @test isnothing(connectivity.west)
    @test isnothing(connectivity.top)
    @test isnothing(connectivity.bottom)

    # +---+
    # | 3 |
    # +---+
    # | 2 |
    # +---+
    # | 1 |
    # +---+
    # | 0 |
    # +---+

    if local_rank == 0
        @test connectivity.north == 1
        @test connectivity.south == 3
    elseif local_rank == 1
        @test connectivity.north == 2
        @test connectivity.south == 0
    elseif local_rank == 2
        @test connectivity.north == 3
        @test connectivity.south == 1
    elseif local_rank == 3
        @test connectivity.north == 0
        @test connectivity.south == 2
    end

    return nothing
end

function test_triply_periodic_rank_connectivity_with_114_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularRectilinearGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(1, 1, 4))

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    @test local_rank == index2rank(arch.local_index..., arch.ranks...)

    connectivity = arch.connectivity

    # No communication in x and y.
    @test isnothing(connectivity.east)
    @test isnothing(connectivity.west)
    @test isnothing(connectivity.north)
    @test isnothing(connectivity.south)

    #   /---/
    #  / 3 /
    # /---/
    #   /---/
    #  / 2 /
    # /---/
    #   /---/
    #  / 1 /
    # /---/
    #   /---/
    #  / 0 /
    # /---/

    if local_rank == 0
        @test connectivity.top == 1
        @test connectivity.bottom == 3
    elseif local_rank == 1
        @test connectivity.top == 2
        @test connectivity.bottom == 0
    elseif local_rank == 2
        @test connectivity.top == 3
        @test connectivity.bottom == 1
    elseif local_rank == 3
        @test connectivity.top == 0
        @test connectivity.bottom == 2
    end

    return nothing
end

function test_triply_periodic_rank_connectivity_with_221_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularRectilinearGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(2, 2, 1))

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    @test local_rank == index2rank(arch.local_index..., arch.ranks...)

    connectivity = arch.connectivity

    # No communication in z.
    @test isnothing(connectivity.top)
    @test isnothing(connectivity.bottom)

    # +---+---+
    # | 0 | 2 |
    # +---+---+
    # | 1 | 3 |
    # +---+---+

    if local_rank == 0
        @test connectivity.east == 2
        @test connectivity.west == 2
        @test connectivity.north == 1
        @test connectivity.south == 1
    elseif local_rank == 1
        @test connectivity.east == 3
        @test connectivity.west == 3
        @test connectivity.north == 0
        @test connectivity.south == 0
    elseif local_rank == 2
        @test connectivity.east == 0
        @test connectivity.west == 0
        @test connectivity.north == 3
        @test connectivity.south == 3
    elseif local_rank == 3
        @test connectivity.east == 1
        @test connectivity.west == 1
        @test connectivity.north == 2
        @test connectivity.south == 2
    end

    return nothing
end

#####
##### Local grids for distributed models
#####

function test_triply_periodic_local_grid_with_411_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularRectilinearGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(4, 1, 1))
    model = DistributedIncompressibleModel(architecture=arch, grid=full_grid, pressure_solver=nothing)

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    local_grid = model.grid
    nx, ny, nz = size(local_grid)

    @test local_grid.xF[1] == 0.25*local_rank
    @test local_grid.xF[nx+1] == 0.25*(local_rank+1)
    @test local_grid.yF[1] == 0
    @test local_grid.yF[ny+1] == 2
    @test local_grid.zF[1] == -3
    @test local_grid.zF[nz+1] == 0

    return nothing
end

function test_triply_periodic_local_grid_with_141_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularRectilinearGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(1, 4, 1))
    model = DistributedIncompressibleModel(architecture=arch, grid=full_grid, pressure_solver=nothing)

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    local_grid = model.grid
    nx, ny, nz = size(local_grid)

    @test local_grid.xF[1] == 0
    @test local_grid.xF[nx+1] == 1
    @test local_grid.yF[1] == 0.5*local_rank
    @test local_grid.yF[ny+1] == 0.5*(local_rank+1)
    @test local_grid.zF[1] == -3
    @test local_grid.zF[nz+1] == 0

    return nothing
end

function test_triply_periodic_local_grid_with_114_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularRectilinearGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(1, 1, 4))
    model = DistributedIncompressibleModel(architecture=arch, grid=full_grid, pressure_solver=nothing)

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    local_grid = model.grid
    nx, ny, nz = size(local_grid)

    @test local_grid.xF[1] == 0
    @test local_grid.xF[nx+1] == 1
    @test local_grid.yF[1] == 0
    @test local_grid.yF[ny+1] == 2
    @test local_grid.zF[1] == -3 + 0.75*local_rank
    @test local_grid.zF[nz+1] == -3 + 0.75*(local_rank+1)

    return nothing
end

function test_triply_periodic_local_grid_with_221_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularRectilinearGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(2, 2, 1))
    model = DistributedIncompressibleModel(architecture=arch, grid=full_grid, pressure_solver=nothing)

    i, j, k = arch.local_index
    local_grid = model.grid
    nx, ny, nz = size(local_grid)

    @test local_grid.xF[1] == 0.5*(i-1)
    @test local_grid.xF[nx+1] == 0.5*i
    @test local_grid.yF[1] == j-1
    @test local_grid.yF[ny+1] == j
    @test local_grid.zF[1] == -3
    @test local_grid.zF[nz+1] == 0

    return nothing
end

#####
##### Injection of halo communication BCs
#####

function test_triply_periodic_bc_injection_with_411_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularRectilinearGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(4, 1, 1))
    model = DistributedIncompressibleModel(architecture=arch, grid=full_grid, pressure_solver=nothing)

    for field in merge(fields(model), model.pressures)
        fbcs = field.boundary_conditions
        @test fbcs.east isa HaloCommunicationBC
        @test fbcs.west isa HaloCommunicationBC
        @test !isa(fbcs.north, HaloCommunicationBC)
        @test !isa(fbcs.south, HaloCommunicationBC)
        @test !isa(fbcs.top, HaloCommunicationBC)
        @test !isa(fbcs.bottom, HaloCommunicationBC)
    end
end

function test_triply_periodic_bc_injection_with_141_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularRectilinearGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(1, 4, 1))
    model = DistributedIncompressibleModel(architecture=arch, grid=full_grid, pressure_solver=nothing)

    for field in merge(fields(model), model.pressures)
        fbcs = field.boundary_conditions
        @test !isa(fbcs.east, HaloCommunicationBC)
        @test !isa(fbcs.west, HaloCommunicationBC)
        @test fbcs.north isa HaloCommunicationBC
        @test fbcs.south isa HaloCommunicationBC
        @test !isa(fbcs.top, HaloCommunicationBC)
        @test !isa(fbcs.bottom, HaloCommunicationBC)
    end
end

function test_triply_periodic_bc_injection_with_114_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularRectilinearGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(1, 1, 4))
    model = DistributedIncompressibleModel(architecture=arch, grid=full_grid, pressure_solver=nothing)

    for field in merge(fields(model), model.pressures)
        fbcs = field.boundary_conditions
        @test !isa(fbcs.east, HaloCommunicationBC)
        @test !isa(fbcs.west, HaloCommunicationBC)
        @test !isa(fbcs.north, HaloCommunicationBC)
        @test !isa(fbcs.south, HaloCommunicationBC)
        @test fbcs.top isa HaloCommunicationBC
        @test fbcs.bottom isa HaloCommunicationBC
    end
end

function test_triply_periodic_bc_injection_with_221_ranks()
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularRectilinearGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
    arch = MultiCPU(grid=full_grid, ranks=(2, 2, 1))
    model = DistributedIncompressibleModel(architecture=arch, grid=full_grid, pressure_solver=nothing)

    for field in merge(fields(model), model.pressures)
        fbcs = field.boundary_conditions
        @test fbcs.east isa HaloCommunicationBC
        @test fbcs.west isa HaloCommunicationBC
        @test fbcs.north isa HaloCommunicationBC
        @test fbcs.south isa HaloCommunicationBC
        @test !isa(fbcs.top, HaloCommunicationBC)
        @test !isa(fbcs.bottom, HaloCommunicationBC)
    end
end

#####
##### Halo communication
#####

function test_triply_periodic_halo_communication_with_411_ranks(halo)
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularRectilinearGrid(topology=topo, size=(16, 6, 4), extent=(1, 2, 3), halo=halo)
    arch = MultiCPU(grid=full_grid, ranks=(4, 1, 1))
    model = DistributedIncompressibleModel(architecture=arch, grid=full_grid, pressure_solver=nothing)

    for field in merge(fields(model), model.pressures)
        @test architecture(field) isa AbstractMultiArchitecture

        interior(field) .= arch.local_rank
        fill_halo_regions!(field)

        @test all(east_halo(field, include_corners=false) .== arch.connectivity.east)
        @test all(west_halo(field, include_corners=false) .== arch.connectivity.west)

        @test all(interior(field) .== arch.local_rank)
        @test all(north_halo(field, include_corners=false) .== arch.local_rank)
        @test all(south_halo(field, include_corners=false) .== arch.local_rank)
        @test all(top_halo(field, include_corners=false) .== arch.local_rank)
        @test all(bottom_halo(field, include_corners=false) .== arch.local_rank)
end


    return nothing
end

function test_triply_periodic_halo_communication_with_141_ranks(halo)
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularRectilinearGrid(topology=topo, size=(4, 16, 4), extent=(1, 2, 3), halo=halo)
    arch = MultiCPU(grid=full_grid, ranks=(1, 4, 1))
    model = DistributedIncompressibleModel(architecture=arch, grid=full_grid, pressure_solver=nothing)

    for field in merge(fields(model), model.pressures)
        interior(field) .= arch.local_rank
        fill_halo_regions!(field)

        @test all(north_halo(field, include_corners=false) .== arch.connectivity.north)
        @test all(south_halo(field, include_corners=false) .== arch.connectivity.south)

        @test all(interior(field) .== arch.local_rank)
        @test all(east_halo(field, include_corners=false) .== arch.local_rank)
        @test all(west_halo(field, include_corners=false) .== arch.local_rank)
        @test all(top_halo(field, include_corners=false) .== arch.local_rank)
        @test all(bottom_halo(field, include_corners=false) .== arch.local_rank)
    end

    return nothing
end

function test_triply_periodic_halo_communication_with_114_ranks(halo)
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularRectilinearGrid(topology=topo, size=(4, 4, 16), extent=(1, 2, 3), halo=halo)
    arch = MultiCPU(grid=full_grid, ranks=(1, 1, 4))
    model = DistributedIncompressibleModel(architecture=arch, grid=full_grid, pressure_solver=nothing)

    for field in merge(fields(model), model.pressures)
        interior(field) .= arch.local_rank
        fill_halo_regions!(field)

        @test all(top_halo(field, include_corners=false) .== arch.connectivity.top)
        @test all(bottom_halo(field, include_corners=false) .== arch.connectivity.bottom)

        @test all(interior(field) .== arch.local_rank)
        @test all(east_halo(field, include_corners=false) .== arch.local_rank)
        @test all(west_halo(field, include_corners=false) .== arch.local_rank)
        @test all(north_halo(field, include_corners=false) .== arch.local_rank)
        @test all(south_halo(field, include_corners=false) .== arch.local_rank)
    end

    return nothing
end

function test_triply_periodic_halo_communication_with_221_ranks(halo)
    topo = (Periodic, Periodic, Periodic)
    full_grid = RegularRectilinearGrid(topology=topo, size=(8, 8, 3), extent=(1, 2, 3), halo=halo)
    arch = MultiCPU(grid=full_grid, ranks=(2, 2, 1))
    model = DistributedIncompressibleModel(architecture=arch, grid=full_grid, pressure_solver=nothing)

    for field in merge(fields(model), model.pressures)
        interior(field) .= arch.local_rank
        fill_halo_regions!(field)

        @test all(east_halo(field, include_corners=false) .== arch.connectivity.east)
        @test all(west_halo(field, include_corners=false) .== arch.connectivity.west)
        @test all(north_halo(field, include_corners=false) .== arch.connectivity.north)
        @test all(south_halo(field, include_corners=false) .== arch.connectivity.south)

        @test all(interior(field) .== arch.local_rank)
        @test all(top_halo(field, include_corners=false) .== arch.local_rank)
        @test all(bottom_halo(field, include_corners=false) .== arch.local_rank)
    end

    return nothing
end

#####
##### Run tests!
#####

@testset "Distributed MPI Oceananigans" begin

    @info "Testing distributed MPI Oceananigans..."

    @testset "Multi architectures rank connectivity" begin
        @info "  Testing multi architecture rank connectivity..."
        test_triply_periodic_rank_connectivity_with_411_ranks()
        test_triply_periodic_rank_connectivity_with_141_ranks()
        test_triply_periodic_rank_connectivity_with_114_ranks()
        test_triply_periodic_rank_connectivity_with_221_ranks()
    end

    @testset "Local grids for distributed models" begin
        @info "  Testing local grids for distributed models..."
        test_triply_periodic_local_grid_with_411_ranks()
        test_triply_periodic_local_grid_with_141_ranks()
        test_triply_periodic_local_grid_with_114_ranks()
        test_triply_periodic_local_grid_with_221_ranks()
    end

    @testset "Injection of halo communication BCs" begin
        @info "  Testing injection of halo communication BCs..."
        test_triply_periodic_bc_injection_with_411_ranks()
        test_triply_periodic_bc_injection_with_141_ranks()
        test_triply_periodic_bc_injection_with_114_ranks()
        test_triply_periodic_bc_injection_with_221_ranks()
    end

    @testset "Halo communication" begin
        @info "  Testing halo communication..."
        for H in 1:3
            test_triply_periodic_halo_communication_with_411_ranks((H, H, H))
            test_triply_periodic_halo_communication_with_141_ranks((H, H, H))
            test_triply_periodic_halo_communication_with_114_ranks((H, H, H))
            test_triply_periodic_halo_communication_with_221_ranks((H, H, H))
        end
    end

    @testset "Time stepping IncompressibleModel" begin
        topo = (Periodic, Periodic, Periodic)
        full_grid = RegularRectilinearGrid(topology=topo, size=(8, 8, 8), extent=(1, 2, 3))
        arch = MultiCPU(grid=full_grid, ranks=(1, 4, 1))
        model = DistributedIncompressibleModel(architecture=arch, grid=full_grid)

        time_step!(model, 1)
        @test model isa IncompressibleModel
        @test model.clock.time ≈ 1

        simulation = Simulation(model, Δt=1, stop_iteration=2)
        run!(simulation)
        @test model isa IncompressibleModel
        @test model.clock.time ≈ 2
    end

    @testset "Time stepping ShallowWaterModel" begin
        topo = (Periodic, Periodic, Flat)
        full_grid = RegularRectilinearGrid(topology=topo, size=(8, 8), extent=(1, 2), halo=(3,3))
        arch = MultiCPU(grid=full_grid, ranks=(1, 4, 1))
        model = DistributedShallowWaterModel(architecture=arch, grid=full_grid, gravitational_acceleration=1)

        set!(model, h=1)
        time_step!(model, 1)
        @test model isa ShallowWaterModel
        @test model.clock.time ≈ 1

        simulation = Simulation(model, Δt=1, stop_iteration=2)
        run!(simulation)
        @test model isa ShallowWaterModel
        @test model.clock.time ≈ 2
    end
end
