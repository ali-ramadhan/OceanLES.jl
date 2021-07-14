"""
    struct DiscreteBoundaryFunction{P, F} <: Function

A wrapper for boundary condition functions with optional parameters.
When `parameters=nothing`, the boundary condition `func` is called with the signature

    `func(i, j, grid, clock, model_fields)`

where `i, j` are the indices along the boundary,
where `grid` is `model.grid`, `clock.time` is the current simulation time and
`clock.iteration` is the current model iteration, and
`model_fields` is a `NamedTuple` with `u, v, w`, the fields in `model.tracers`,
and the fields in `model.diffusivities`, each of which is an `OffsetArray`s (or `NamedTuple`s
of `OffsetArray`s depending on the turbulence closure) of field data.

When `parameters` is not `nothing`, the boundary condition `func` is called with
the signature

    `func(i, j, grid, clock, model_fields)`

*Note* that the index `end` does *not* access the final physical grid point of
a model field in any direction. The final grid point must be explictly specified, as
in `model_fields.u[i, j, grid.Nz]`.
"""
struct DiscreteBoundaryFunction{P, F} <: Function
    func :: F
    parameters :: P
end

# Un-parameterized
@inline (bc::DiscreteBoundaryFunction{<:Nothing})(i, j, grid, clock, model_fields) =
    bc.func(i, j, grid, clock, model_fields)

# Parameterized
@inline (bc::DiscreteBoundaryFunction)(i, j, grid, clock, model_fields) =
    bc.func(i, j, grid, clock, model_fields, bc.parameters)

# Don't re-convert DiscreteBoundaryFunctions passed to BoundaryCondition constructor
BoundaryCondition(Classification::DataType, condition::DiscreteBoundaryFunction) = BoundaryCondition(Classification(), condition)

Adapt.adapt_structure(to, bf::DiscreteBoundaryFunction) = DiscreteBoundaryFunction(Adapt.adapt(to, bf.func),
                                                                                   Adapt.adapt(to, bf.parameters))
