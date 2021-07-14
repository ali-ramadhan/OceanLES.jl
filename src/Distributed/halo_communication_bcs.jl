using Oceananigans.BoundaryConditions
using Oceananigans.BoundaryConditions: AbstractBoundaryConditionClassification

import Oceananigans.BoundaryConditions: bc_str, print_condition

struct HaloCommunication <: AbstractBoundaryConditionClassification end

const HaloCommunicationBC = BoundaryCondition{<:HaloCommunication}

bc_str(::HaloCommunicationBC) = "HaloCommunication"

HaloCommunicationBoundaryCondition(val; kwargs...) = BoundaryCondition(HaloCommunication, val; kwargs...)

struct HaloCommunicationRanks{F, T}
    from :: F
      to :: T
end

HaloCommunicationRanks(; from, to) = HaloCommunicationRanks(from, to)

print_condition(hcr::HaloCommunicationRanks) = "(from rank $(hcr.from) to rank $(hcr.to))"

function inject_halo_communication_boundary_conditions(field_bcs, local_rank, connectivity)
    rank_east = connectivity.east
    rank_west = connectivity.west
    rank_north = connectivity.north
    rank_south = connectivity.south
    rank_top = connectivity.top
    rank_bottom = connectivity.bottom

    east_comm_ranks = HaloCommunicationRanks(from=local_rank, to=rank_east)
    west_comm_ranks = HaloCommunicationRanks(from=local_rank, to=rank_west)
    north_comm_ranks = HaloCommunicationRanks(from=local_rank, to=rank_north)
    south_comm_ranks = HaloCommunicationRanks(from=local_rank, to=rank_south)
    top_comm_ranks = HaloCommunicationRanks(from=local_rank, to=rank_top)
    bottom_comm_ranks = HaloCommunicationRanks(from=local_rank, to=rank_bottom)

    east_comm_bc = HaloCommunicationBoundaryCondition(east_comm_ranks)
    west_comm_bc = HaloCommunicationBoundaryCondition(west_comm_ranks)
    north_comm_bc = HaloCommunicationBoundaryCondition(north_comm_ranks)
    south_comm_bc = HaloCommunicationBoundaryCondition(south_comm_ranks)
    top_comm_bc = HaloCommunicationBoundaryCondition(top_comm_ranks)
    bottom_comm_bc = HaloCommunicationBoundaryCondition(bottom_comm_ranks)

    west = isnothing(rank_west) ? field_bcs.west : west_comm_bc
    east = isnothing(rank_east) ? field_bcs.east : east_comm_bc
    south = isnothing(rank_south) ? field_bcs.south : south_comm_bc
    north = isnothing(rank_north) ? field_bcs.north : north_comm_bc
    bottom = isnothing(rank_bottom) ? field_bcs.bottom : bottom_comm_bc
    top = isnothing(rank_top) ? field_bcs.top : top_comm_bc
    immersed = field_bcs.immersed

    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end
