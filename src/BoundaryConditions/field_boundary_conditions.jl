using Oceananigans.Operators: assumed_field_location

#####
##### Default boundary conditions
#####

struct DefaultPrognosticFieldBoundaryCondition end

default_prognostic_field_boundary_condition(::Grids.Periodic, loc)       = PeriodicBoundaryCondition()
default_prognostic_field_boundary_condition(::Bounded,        ::Center)  = NoFluxBoundaryCondition()
default_prognostic_field_boundary_condition(::Bounded,        ::Face)    = ImpenetrableBoundaryCondition()

default_prognostic_field_boundary_condition(::Bounded,        ::Nothing) = nothing
default_prognostic_field_boundary_condition(::Flat,           ::Nothing) = nothing
default_prognostic_field_boundary_condition(::Grids.Periodic, ::Nothing) = nothing
default_prognostic_field_boundary_condition(::Flat, loc) = nothing

default_auxiliary_field_boundary_condition(topo, loc) = default_prognostic_field_boundary_condition(topo, loc)
default_auxiliary_field_boundary_condition(::Bounded, ::Face) = nothing

#####
##### Field boundary conditions
#####

struct FieldBoundaryConditions{W, E, S, N, B, T, I}
        west :: W
        east :: E
       south :: S
       north :: N
      bottom :: B
         top :: T
    immersed :: I
end

"""
    FieldBoundaryConditions(; kwargs...)

Returns a template for boundary conditions on prognostic fields.

Keyword arguments specify boundary conditions on the 7 possible boundaries:

    * `west`, left end point in the `x`-direction where `i=1`
    * `east`, right end point in the `x`-direction where `i=grid.Nx`
    * `south`, left end point in the `y`-direction where `j=1`
    * `north`, right end point in the `y`-direction where `j=grid.Ny`
    * `bottom`, right end point in the `z`-direction where `k=1`
    * `top`, right end point in the `z`-direction where `k=grid.Nz`
    * `immersed`, boundary between solid and fluid for immersed boundaries (experimental support only)

If a boundary condition is unspecified, the default for prognostic fields
and the topology in the boundary-normal direction is used:

    * `PeriodicBoundaryCondition` for `Periodic` directions
    * `NoFluxBoundaryCondition` for `Bounded` directions and `Centered`-located fields
    * `ImpenetrableBoundaryCondition` for `Bounded` directions and `Face`-located fields
    * `nothing` for `Flat` directions and/or `Nothing`-located fields
"""
function FieldBoundaryConditions(;     west = DefaultPrognosticFieldBoundaryCondition(),
                                       east = DefaultPrognosticFieldBoundaryCondition(),
                                      south = DefaultPrognosticFieldBoundaryCondition(),
                                      north = DefaultPrognosticFieldBoundaryCondition(),
                                     bottom = DefaultPrognosticFieldBoundaryCondition(),
                                        top = DefaultPrognosticFieldBoundaryCondition(),
                                   immersed = NoFluxBoundaryCondition())

   return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

"""
    FieldBoundaryConditions(grid, location; kwargs...)

Returns boundary conditions for auxiliary fields (fields
whose values are derived from a model's prognostic fields) on `grid`
and at `location`.

Keyword arguments specify boundary conditions on the 6 possible boundaries:

    * `west`, left end point in the `x`-direction where `i=1`
    * `east`, right end point in the `x`-direction where `i=grid.Nx`
    * `south`, left end point in the `y`-direction where `j=1`
    * `north`, right end point in the `y`-direction where `j=grid.Ny`
    * `bottom`, right end point in the `z`-direction where `k=1`
    * `top`, right end point in the `z`-direction where `k=grid.Nz`

If a boundary condition is unspecified, the default for auxiliary fields
and the topology in the boundary-normal direction is used:

    * `PeriodicBoundaryCondition` for `Periodic` directions
    * `GradientBoundaryCondition(0)` for `Bounded` directions and `Centered`-located fields
    * `nothing` for `Bounded` directions and `Face`-located fields
    * `nothing` for `Flat` directions and/or `Nothing`-located fields)
"""
function FieldBoundaryConditions(grid, loc;
                                   west = default_auxiliary_field_boundary_condition(topology(grid, 1)(), loc[1]()),
                                   east = default_auxiliary_field_boundary_condition(topology(grid, 1)(), loc[1]()),
                                  south = default_auxiliary_field_boundary_condition(topology(grid, 2)(), loc[2]()),
                                  north = default_auxiliary_field_boundary_condition(topology(grid, 2)(), loc[2]()),
                                 bottom = default_auxiliary_field_boundary_condition(topology(grid, 3)(), loc[3]()),
                                    top = default_auxiliary_field_boundary_condition(topology(grid, 3)(), loc[3]()),
                               immersed = NoFluxBoundaryCondition())

   return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

#####
##### Boundary condition "regularization"
#####

regularize_boundary_condition(::DefaultPrognosticFieldBoundaryCondition, topo, loc, dim, args...) =
    default_prognostic_field_boundary_condition(topo[dim](), loc[dim]())

regularize_boundary_condition(bc, args...) = bc # fallback

""" 
Compute default boundary conditions and attach field locations to ContinuousBoundaryFunction
boundary conditions for prognostic model field boundary conditions.

Note: don't regularize immersed boundary conditions: we don't support ContinuousBoundaryFunction
for immersed boundary conditions.
"""
function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions, grid, field_name, prognostic_field_names=nothing)

    topo = topology(grid)
    loc = assumed_field_location(field_name)
    
    west     = regularize_boundary_condition(bcs.west,   topo, loc, 1, 1,       prognostic_field_names)
    east     = regularize_boundary_condition(bcs.east,   topo, loc, 1, grid.Nx, prognostic_field_names)
    south    = regularize_boundary_condition(bcs.south,  topo, loc, 2, 1,       prognostic_field_names)
    north    = regularize_boundary_condition(bcs.north,  topo, loc, 2, grid.Ny, prognostic_field_names)
    bottom   = regularize_boundary_condition(bcs.bottom, topo, loc, 3, 1,       prognostic_field_names)
    top      = regularize_boundary_condition(bcs.top,    topo, loc, 3, grid.Nz, prognostic_field_names)

    # Eventually we could envision supporting ContinuousForcing-style boundary conditions
    # for the immersed boundary condition, which would benefit from regularization.
    # But for now we don't regularize the immersed boundary condition.
    immersed = bcs.immersed

    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

regularize_field_boundary_conditions(boundary_conditions::NamedTuple, grid, prognostic_field_names) =
    NamedTuple(field_name => regularize_field_boundary_conditions(field_bcs, grid, field_name, prognostic_field_names)
               for (field_name, field_bcs) in pairs(boundary_conditions))

# For nested NamedTuples of boundary conditions (eg diffusivity boundary conditions)
regularize_field_boundary_conditions(boundary_conditions::NamedTuple, grid, ::Symbol, prognostic_field_names) =
    NamedTuple(field_name => regularize_field_boundary_conditions(field_bcs, grid, field_name, prognostic_field_names)
               for (field_name, field_bcs) in pairs(boundary_conditions))

regularize_field_boundary_conditions(::Missing, grid, field_name, prognostic_field_names=nothing) = missing
