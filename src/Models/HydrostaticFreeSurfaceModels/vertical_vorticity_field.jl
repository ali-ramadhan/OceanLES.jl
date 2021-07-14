using Oceananigans.Grids: Face, Face, Center
using Oceananigans.Operators: ζ₃ᶠᶠᵃ
using Oceananigans.Fields: KernelComputedField
using KernelAbstractions: @kernel

@kernel function compute_vertical_vorticity!(ζ, grid, u, v)
    i, j, k, = @index(Global, NTuple)
    @inbounds ζ[i, j, k] = ζ₃ᶠᶠᵃ(i, j, k, grid, u, v)
end

"""
    VerticalVorticityField(model)

Returns a KernelComputedField that `compute!`s vertical vorticity in a
manner consistent with the `VectorInvariant` momentum advection scheme
for curvilinear grids.

In particular, `VerticalVorticityField` uses `ζ₃ᶠᶠᵃ`, which in turn computes
the vertical vorticity by first integrating the velocity field around the borders
of the vorticity cell to find the vertical circulation, and then dividing by the area of
the vorticity cell to compute vertical vorticity.
"""
VerticalVorticityField(model) = KernelComputedField(Face, Face, Center, compute_vertical_vorticity!, model, 
                                                    computed_dependencies = (model.velocities.u, model.velocities.v))
