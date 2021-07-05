using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields: validate_field_data

import Oceananigans.Fields: Field

function Field(X, Y, Z, arch::AbstractMultiArchitecture, grid::AbstractGrid,
                bcs = FieldBoundaryConditions(grid, (X, Y, Z)),
               data = new_data(eltype(grid), arch, grid, (X, Y, Z)))

    communicative_bcs = inject_halo_communication_boundary_conditions(bcs, arch.local_rank, arch.connectivity)
    validate_field_data(X, Y, Z, data, grid)

    return Field{X, Y, Z}(data, arch, grid, bcs)
end
