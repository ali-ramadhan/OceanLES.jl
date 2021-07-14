using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: XFaceField, YFaceField, ZFaceField, TracerFields

function HydrostaticFreeSurfaceVelocityFields(::Nothing, arch, grid, clock, bcs=NamedTuple())
    u = XFaceField(arch, grid, bcs.u)
    v = YFaceField(arch, grid, bcs.v)
    w = ZFaceField(arch, grid)

    return (u=u, v=v, w=w)
end

function HydrostaticFreeSurfaceTendencyFields(velocities, free_surface, arch, grid, tracer_names)
    u = XFaceField(arch, grid)
    v = YFaceField(arch, grid)
    η = FreeSurfaceDisplacementField(velocities, free_surface, arch, grid)
    tracers = TracerFields(tracer_names, arch, grid)

    return merge((u=u, v=v, η=η), tracers)
end
