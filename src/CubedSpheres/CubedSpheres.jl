module CubedSpheres

export ConformalCubedSphereGrid, face, faces, λnodes, φnodes

include("cubed_sphere_utils.jl")
include("conformal_cubed_sphere_grid.jl")
include("cubed_sphere_exchange_bcs.jl")
include("cubed_sphere_faces.jl")
include("cubed_sphere_set!.jl")
include("cubed_sphere_halo_filling.jl")
include("cubed_sphere_kernel_launching.jl")
include("immersed_conformal_cubed_sphere_grid.jl")

#####
##### Validating cubed sphere stuff
#####

import Oceananigans.Fields: validate_field_data
import Oceananigans.Models.HydrostaticFreeSurfaceModels: validate_vertical_velocity_boundary_conditions

function validate_field_data(X, Y, Z, data, grid::ConformalCubedSphereGrid)

    for (face_data, face_grid) in zip(data.faces, grid.faces)
        validate_field_data(X, Y, Z, face_data, face_grid)
    end

    return nothing
end

validate_vertical_velocity_boundary_conditions(w::AbstractCubedSphereField) =
    [validate_vertical_velocity_boundary_conditions(w_face) for w_face in faces(w)]

#####
##### Regularizing field boundary conditions
#####

import Oceananigans.BoundaryConditions: regularize_field_boundary_conditions

function regularize_field_boundary_conditions(bcs::CubedSphereFaces, grid, field_name, prognostic_field_names)

    faces = Tuple(regularize_field_boundary_conditions(face_bcs, face_grid, field_name, prognostic_field_names)
                  for (face_bcs, face_grid) in zip(bcs.faces, grid.faces))

    return CubedSphereFaces{typeof(faces[1]), typeof(faces)}(faces)
end

function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions, grid::ConformalCubedSphereGrid, field_name, prognostic_field_names)

    faces = Tuple(
        inject_cubed_sphere_exchange_boundary_conditions(
            regularize_field_boundary_conditions(bcs, face_grid, field_name, prognostic_field_names),
            face_index,
            grid.face_connectivity
        )
        for (face_index, face_grid) in enumerate(grid.faces)
    )

    return CubedSphereFaces{typeof(faces[1]), typeof(faces)}(faces)
end

#####
##### Applying flux boundary conditions
#####

import Oceananigans.Models.HydrostaticFreeSurfaceModels: apply_flux_bcs!

function apply_flux_bcs!(Gcⁿ::AbstractCubedSphereField, events, c::AbstractCubedSphereField, arch, barrier, clock, model_fields)

    for (face_index, Gcⁿ_face) in enumerate(faces(Gcⁿ))
        apply_flux_bcs!(Gcⁿ_face, events, get_face(c, face_index), arch, barrier,
                        clock, get_face(model_fields, face_index))
    end

    return nothing
end

#####
##### Forcing functions on the cubed sphere
#####

using Oceananigans.Forcings: user_function_arguments
import Oceananigans.Forcings: ContinuousForcing

@inline function (forcing::ContinuousForcing{LX, LY, LZ})(i, j, k, grid::ConformalCubedSphereFaceGrid, clock, model_fields) where {LX, LY, LZ}

    args = user_function_arguments(i, j, k, grid, model_fields, forcing.parameters, forcing)

    λ = λnode(LX(), LY(), LZ(), i, j, k, grid)
    φ = φnode(LX(), LY(), LZ(), i, j, k, grid)
    z = znode(LX(), LY(), LZ(), i, j, k, grid)

    return @inbounds forcing.func(λ, φ, z, clock.time, args...)
end

#####
##### NaN checker for cubed sphere fields
#####

import Oceananigans.Diagnostics: error_if_nan_in_field

function error_if_nan_in_field(field::AbstractCubedSphereField, name, clock)
    for (face_index, face_field) in enumerate(faces(field))
        error_if_nan_in_field(face_field, string(name) * " (face $face_index)", clock)
    end
end

#####
##### CFL for cubed sphere fields
#####

import Oceananigans.Diagnostics: accurate_cell_advection_timescale

function accurate_cell_advection_timescale(grid::ConformalCubedSphereGrid, velocities)

    min_timescale_on_faces = []

    for (face_index, grid_face) in enumerate(grid.faces)
        velocities_face = get_face(velocities, face_index)
        min_timescale_on_face = accurate_cell_advection_timescale(grid_face, velocities_face)
        push!(min_timescale_on_faces, min_timescale_on_face)
    end

    return minimum(min_timescale_on_faces)
end

#####
##### Output writing for cubed sphere fields
#####

using Oceananigans.Fields: compute!
import Oceananigans.OutputWriters: fetch_output

function fetch_output(field::AbstractCubedSphereField, model, field_slicer)
    compute!(field)
    return Tuple(fetch_output(face_field, model, field_slicer) for face_field in faces(field))
end

#####
##### StateChecker for each face is useful for debugging
#####

import Oceananigans.Diagnostics: state_check

function state_check(field::AbstractCubedSphereField, name, pad)
    face_fields = faces(field)
    Nf = length(face_fields)
    for (face_index, face_field) in enumerate(face_fields)
        face_str = " face $face_index"
        state_check(face_field, string(name) * face_str, pad + length(face_str))

        # Leave empty line between fields for easier visual inspection.
        face_index == Nf && @info ""
    end
end

end # module
