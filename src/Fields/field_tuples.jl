using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions

#####
##### Velocity fields tuples
#####

"""
    VelocityFields(arch, grid, bcs::NamedTuple)

Return a NamedTuple with fields `u`, `v`, `w` initialized on the architecture `arch`
and `grid`. Boundary conditions `bcs` may be specified via a named tuple of
`FieldBoundaryCondition`s.
"""
function VelocityFields(arch, grid, user_bcs = NamedTuple())

    template = FieldBoundaryConditions()

    default_bcs = (
        u = regularize_field_boundary_conditions(template, grid, :u),
        v = regularize_field_boundary_conditions(template, grid, :v),
        w = regularize_field_boundary_conditions(template, grid, :w)
    )

    bcs = merge(default_bcs, user_bcs)

    u = XFaceField(arch, grid, bcs.u)
    v = YFaceField(arch, grid, bcs.v)
    w = ZFaceField(arch, grid, bcs.w)

    return (u=u, v=v, w=w)
end

#####
##### Tracer fields tuples
#####

"""
    TracerFields(tracer_names, arch, grid, bcs)

Returns a `NamedTuple` with tracer fields specified by `tracer_names` initialized as
`CenterField`s on the architecture `arch` and `grid`. Boundary conditions `bcs` may
be specified via a named tuple of `FieldBoundaryCondition`s.
"""
function TracerFields(tracer_names, arch, grid, user_bcs)
    default_bcs = NamedTuple(name => FieldBoundaryConditions(grid, (Center, Center, Center)) for name in tracer_names)
    bcs = merge(default_bcs, user_bcs) # provided bcs overwrite defaults
    return NamedTuple(c => CenterField(arch, grid, bcs[c]) for c in tracer_names)
end

"""
    TracerFields(tracer_names, arch, grid; kwargs...)

Return a NamedTuple with tracer fields specified by `tracer_names` initialized as
`CenterField`s on the architecture `arch` and `grid`. Fields may be passed via optional
keyword arguments `kwargs` for each field.

This function is used by `OutputWriters.Checkpointer` and `TendencyFields`.
```
"""
TracerFields(tracer_names, arch, grid; kwargs...) =
    NamedTuple(c => c ∈ keys(kwargs) ? kwargs[c] : CenterField(arch, grid) for c in tracer_names)

# 'Nothing', or empty tracer fields
TracerFields(::Union{Tuple{}, Nothing}, arch, grid, bcs) = NamedTuple()

"Shortcut constructor for empty tracer fields."
TracerFields(::NamedTuple{(), Tuple{}}, arch, grid, bcs) = NamedTuple()

#####
##### Pressure fields tuples
#####

"""
    PressureFields(arch, grid, bcs::NamedTuple)

Return a NamedTuple with pressure fields `pHY′` and `pNHS` initialized as
`CenterField`s on the architecture `arch` and `grid`.  Boundary conditions `bcs` may
be specified via a named tuple of `FieldBoundaryCondition`s.
"""
function PressureFields(arch, grid, bcs=NamedTuple())

    default_pressure_boundary_conditions =
        (pHY′ = FieldBoundaryConditions(grid, (Center, Center, Center)),
         pNHS = FieldBoundaryConditions(grid, (Center, Center, Center)))

    bcs = merge(default_pressure_boundary_conditions, bcs)

    pHY′ = CenterField(arch, grid, bcs.pHY′)
    pNHS = CenterField(arch, grid, bcs.pNHS)

    return (pHY′=pHY′, pNHS=pNHS)
end

"""
    TendencyFields(arch, grid, tracer_names; kwargs...)

Return a NamedTuple with tendencies for all solution fields (velocity fields and
tracer fields), initialized on the architecture `arch` and `grid`. Optional `kwargs`
can be specified to assign data arrays to each tendency field.
"""
function TendencyFields(arch, grid, tracer_names;
                        u = XFaceField(arch, grid),
                        v = YFaceField(arch, grid),
                        w = ZFaceField(arch, grid),
                        kwargs...)

    velocities = (u=u, v=v, w=w)

    tracers = TracerFields(tracer_names, arch, grid; kwargs...)

    return merge(velocities, tracers)
end

#####
##### Helper functions for IncompressibleModel constructor
#####

VelocityFields(::Nothing, arch, grid, bcs) = VelocityFields(arch, grid, bcs)
PressureFields(::Nothing, arch, grid, bcs) = PressureFields(arch, grid, bcs)

"""
    VelocityFields(proposed_velocities::NamedTuple, arch, grid, tracer_names, bcs)

Returns a `NamedTuple` of velocity fields, overwriting boundary conditions
in `proposed_velocities` with corresponding fields in the `NamedTuple` `bcs`.
"""
function VelocityFields(proposed_velocities::NamedTuple{(:u, :v, :w)}, arch, grid, bcs)

    validate_field_tuple_grid("velocities", proposed_velocities, grid)

    u = XFaceField(arch, grid, bcs.u, proposed_velocities.u.data)
    v = YFaceField(arch, grid, bcs.v, proposed_velocities.v.data)
    w = ZFaceField(arch, grid, bcs.w, proposed_velocities.w.data)

    return (u=u, v=v, w=w)
end

"""
    TracerFields(proposed_tracers::NamedTuple, arch, grid, bcs)

Returns a `NamedTuple` of tracers, overwriting boundary conditions
in `proposed_tracers` with corresponding fields in the `NamedTuple` `bcs`.
"""
function TracerFields(proposed_tracers::NamedTuple, arch, grid, bcs)

    validate_field_tuple_grid("tracers", proposed_tracers, grid)

    tracer_names = propertynames(proposed_tracers)
    tracer_fields = Tuple(CenterField(arch, grid, bcs[c], proposed_tracers[c].data) for c in tracer_names)

    return NamedTuple{tracer_names}(tracer_fields)
end

"""
    PressureFields(pressures::NamedTuple, arch, grid, tracer_names, bcs)

Returns a `NamedTuple` of pressure fields with, overwriting boundary conditions
in `proposed_tracer_fields` with corresponding fields in the `NamedTuple` `bcs`.
"""
function PressureFields(proposed_pressures::NamedTuple{(:pHY′, :pNHS)}, arch, grid, bcs)
    validate_field_tuple_grid("pressures", proposed_pressures, grid)

    pHY′ = CenterField(arch, grid, bcs.pHY′, proposed_pressures.pHY′.data)
    pNHS = CenterField(arch, grid, bcs.pNHS, proposed_pressures.pNHS.data)

    return (pHY′=pHY′, pNHS=pNHS)
end
