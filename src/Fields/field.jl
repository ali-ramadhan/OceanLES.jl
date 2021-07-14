using Adapt

struct Field{LX, LY, LZ, A, D, G, T, B} <: AbstractDataField{LX, LY, LZ, A, G, T, 3}
                   data :: D
           architecture :: A
                   grid :: G
    boundary_conditions :: B

    function Field{LX, LY, LZ}(data::D, arch::A, grid::G, bcs::B) where {LX, LY, LZ, D, A, G, B}
        T = eltype(grid)
        return new{LX, LY, LZ, A, D, G, T, B}(data, arch, grid, bcs)
    end
end

"""
    Field(LX, LY, LZ, [arch = CPU()], grid,
          [ bcs = FieldBoundaryConditions(grid, (LX, LY, LZ)),
           data = new_data(eltype(grid), arch, grid, (LX, LY, LZ))])

Construct a `Field` on `grid` with `data` on architecture `arch` with
boundary conditions `bcs`. Each of `(LX, LY, LZ)` is either `Center` or `Face` and determines
the field's location in `(x, y, z)`.

Example
=======

```jldoctest
julia> using Oceananigans

julia> ω = Field(Face, Face, Center, CPU(), RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)))
Field located at (Face, Face, Center)
├── data: OffsetArrays.OffsetArray{Float64, 3, Array{Float64, 3}}, size: (1, 1, 1)
├── grid: RegularRectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=1, Ny=1, Nz=1)
└── boundary conditions: west=Periodic, east=Periodic, south=Periodic, north=Periodic, bottom=ZeroFlux, top=ZeroFlux, immersed=ZeroFlux
```
"""
function Field(LX, LY, LZ,
               arch::AbstractArchitecture,
               grid::AbstractGrid,
               bcs = FieldBoundaryConditions(grid, (LX, LY, LZ)),
               data = new_data(eltype(grid), arch, grid, (LX, LY, LZ)))

    validate_field_data(LX, LY, LZ, data, grid)

    return Field{LX, LY, LZ}(data, arch, grid, bcs)
end

# Default CPU architecture
Field(LX, LY, LZ, grid::AbstractGrid, args...) = Field(LX, LY, LZ, CPU(), grid, args...)

# Canonical `similar` for AbstractField (doesn't transfer boundary conditions)
Base.similar(f::AbstractField{LX, LY, LZ, Arch}) where {LX, LY, LZ, Arch} = Field(LX, LY, LZ, Arch(), f.grid)

# Type "destantiation": convert Face() to Face and Center() to Center if needed.
destantiate(X) = typeof(X)
destantiate(X::DataType) = X

"""
    Field(L::Tuple, arch, grid, data, bcs)

Construct a `Field` at the location defined by the 3-tuple `L`,
whose elements are `Center` or `Face`.
"""
Field(L::Tuple, args...) = Field(destantiate.(L)..., args...)

#####
##### Special constructors for tracers and velocity fields
#####

"""
    CenterField([arch=CPU()], grid, args...)

Returns `Field{Center, Center, Center}` on `arch`itecture and `grid`.
Additional arguments are passed to the `Field` constructor.
"""
CenterField(arch::AbstractArchitecture, grid, args...) = Field(Center, Center, Center, arch, grid, args...)

"""
    XFaceField([arch=CPU()], grid, args...)

Returns `Field{Face, Center, Center}` on `arch`itecture and `grid`.
Additional arguments are passed to the `Field` constructor.
"""
XFaceField(arch::AbstractArchitecture, args...) = Field(Face, Center, Center, arch, args...)

"""
    YFaceField([arch=CPU()], grid, args...)

Returns `Field{Center, Face, Center}` on `arch`itecture and `grid`.
Additional arguments are passed to the `Field` constructor.
"""
YFaceField(arch::AbstractArchitecture, args...) = Field(Center, Face, Center, arch, args...)

"""
    ZFaceField([arch=CPU()], grid, args...)

Returns `Field{Center, Center, Face}` on `arch`itecture and `grid`.
Additional arguments are passed to the `Field` constructor.
"""
ZFaceField(arch::AbstractArchitecture, args...) = Field(Center, Center, Face, arch, args...)

# Default CPU architectures...
CenterField(grid::AbstractGrid, args...) = CenterField(CPU(), grid, args...)
 XFaceField(grid::AbstractGrid, args...) =  XFaceField(CPU(), grid, args...)
 YFaceField(grid::AbstractGrid, args...) =  YFaceField(CPU(), grid, args...)
 ZFaceField(grid::AbstractGrid, args...) =  ZFaceField(CPU(), grid, args...)

Adapt.adapt_structure(to, field::Field) = Adapt.adapt(to, field.data)
