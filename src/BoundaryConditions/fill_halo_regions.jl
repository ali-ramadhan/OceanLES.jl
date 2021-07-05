using OffsetArrays: OffsetArray

#####
##### General halo filling functions
#####

fill_halo_regions!(::Nothing, args...) = []

"""
    fill_halo_regions!(fields, args...)

Fill halo regions for each field in the tuple `fields` according to their boundary
conditions.
"""
function fill_halo_regions!(fields::Union{Tuple, NamedTuple}, args...)

    for field in fields
        fill_halo_regions!(field, args...)
    end

    return nothing
end

# Some fields have `nothing` boundary conditions, such as `FunctionField` and `ZeroField`.
fill_halo_regions!(c::OffsetArray, ::Nothing, args...; kwargs...) = nothing

"Fill halo regions in x, y, and z for a given field's data."
function fill_halo_regions!(c::OffsetArray, bcs, arch, grid, loc, args...; kwargs...)

    barrier = Event(device(arch))

      west_event =   fill_west_halo!(c, bcs.west,   arch, barrier, grid, loc, args...; kwargs...)
      east_event =   fill_east_halo!(c, bcs.east,   arch, barrier, grid, loc, args...; kwargs...)
     south_event =  fill_south_halo!(c, bcs.south,  arch, barrier, grid, loc, args...; kwargs...)
     north_event =  fill_north_halo!(c, bcs.north,  arch, barrier, grid, loc, args...; kwargs...)
    bottom_event = fill_bottom_halo!(c, bcs.bottom, arch, barrier, grid, loc, args...; kwargs...)
       top_event =    fill_top_halo!(c, bcs.top,    arch, barrier, grid, loc, args...; kwargs...)

    # Wait at the end
    events = [west_event, east_event, south_event, north_event, bottom_event, top_event]
    events = filter(e -> e isa Event, events)
    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

#####
##### Halo-filling for nothing boundary conditions
#####

  fill_west_halo!(c, ::Nothing, args...; kwargs...) = nothing
  fill_east_halo!(c, ::Nothing, args...; kwargs...) = nothing
 fill_south_halo!(c, ::Nothing, args...; kwargs...) = nothing
 fill_north_halo!(c, ::Nothing, args...; kwargs...) = nothing
   fill_top_halo!(c, ::Nothing, args...; kwargs...) = nothing
fill_bottom_halo!(c, ::Nothing, args...; kwargs...) = nothing
