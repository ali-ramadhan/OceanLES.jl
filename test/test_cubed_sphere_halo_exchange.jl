using Oceananigans.CubedSpheres
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.CubedSpheres: west_halo, east_halo, south_halo, north_halo, fill_horizontal_velocity_halos!, get_face

# Opposite of the `Base.digits` function
# Source: https://stackoverflow.com/a/55529778
function undigits(d; base=10)
    (s, b) = promote(zero(eltype(d)), base)
    mult = one(s)
    for val in d
        s += val * mult
        mult *= b
    end
    return s
end

cs32_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"

for arch in archs
    @testset "Cubed sphere tracer halo exchange [$(typeof(arch))]" begin
        @info "  Testing cubed sphere tracer halo exchange [$(typeof(arch))]..."

        grid = ConformalCubedSphereGrid(cs32_filepath, architecture=arch, Nz=1, z=(-1, 0))
        field = CenterField(arch, grid)

        ## We will fill each grid point with a 5-digit integer "fiijj" where
        ## the f digit is the face number, the ii digits are the i index, and
        ## the jj digits are the j index. We then check that the halo exchange
        ## happened correctly.

        face_digit(n) = digits(abs(Int(n)))[5]
        i_digits(n) = digits(abs(Int(n)))[3:4] |> undigits
        j_digits(n) = digits(abs(Int(n)))[1:2] |> undigits

        for (face_number, field_face) in enumerate(faces(field))
            for i in 1:field_face.grid.Nx, j in 1:field_face.grid.Ny
                CUDA.@allowscalar field_face[i, j, 1] = parse(Int, @sprintf("%d%02d%02d", face_number, i, j))
            end
        end

        fill_halo_regions!(field)

        CUDA.allowscalar(true)

        @testset "Source and destination faces are correct" begin
            for (face_number, field_face) in enumerate(faces(field))
                west_halo_vals =  west_halo(field_face, include_corners=false) |> Array
                east_halo_vals =  east_halo(field_face, include_corners=false) |> Array
                south_halo_vals = south_halo(field_face, include_corners=false) |> Array
                north_halo_vals = north_halo(field_face, include_corners=false) |> Array

                @test all(face_digit.(west_halo_vals)  .== grid.face_connectivity[face_number].west.face)
                @test all(face_digit.(east_halo_vals)  .== grid.face_connectivity[face_number].east.face)
                @test all(face_digit.(south_halo_vals) .== grid.face_connectivity[face_number].south.face)
                @test all(face_digit.(north_halo_vals) .== grid.face_connectivity[face_number].north.face)
            end
        end

        @testset "1W halo <- 5N boundary halo exchange" begin
            # Grid point [i, j] = [0, 1] in 1W halo should be from [i, j] = [32, 32] in 5N boundary.
            west_halo_south_value = get_face(field, 1)[0, 1, 1]
            @test face_digit(west_halo_south_value) == 5
            @test i_digits(west_halo_south_value) == 32
            @test j_digits(west_halo_south_value) == 32

            # Grid point [i, j] = [0, 32] in 1W halo should be from [i, j] = [1, 32] in 5N boundary.
            west_halo_north_value = get_face(field, 1)[0, 32, 1]
            @test face_digit(west_halo_north_value) == 5
            @test i_digits(west_halo_north_value) == 1
            @test j_digits(west_halo_north_value) == 32

            west_halo_values = west_halo(get_face(field, 1), include_corners=false)[:] |> Array
            @test all(face_digit.(west_halo_values) .== 5)
            @test all(i_digits.(west_halo_values) .== reverse(1:32))
            @test all(j_digits.(west_halo_values) .== 32)
        end

        @testset "1E halo <- 2W boundary halo exchange" begin
            # Grid point [i, j] = [33, 1] in 1E halo should be from [i, j] = [1, 1] in 2W boundary.
            east_halo_south_value = get_face(field, 1)[33, 1, 1]
            @test face_digit(east_halo_south_value) == 2
            @test i_digits(east_halo_south_value) == 1
            @test j_digits(east_halo_south_value) == 1

            # Grid point [i, j] = [33, 32] in 1E halo should be from [i, j] = [1, 32] in 2W boundary.
            east_halo_north_value = get_face(field, 1)[33, 32, 1]
            @test face_digit(east_halo_north_value) == 2
            @test i_digits(east_halo_north_value) == 1
            @test j_digits(east_halo_north_value) == 32

            east_halo_values = east_halo(get_face(field, 1), include_corners=false)[:] |> Array
            @test all(face_digit.(east_halo_values) .== 2)
            @test all(i_digits.(east_halo_values) .== 1)
            @test all(j_digits.(east_halo_values) .== 1:32)
        end

        @testset "1S halo <- 6N boundary halo exchange" begin
            # Grid point [i, j] = [1, 0] in 1S halo should be from [i, j] = [1, 32] in 6N boundary.
            south_halo_west_value = get_face(field, 1)[1, 0, 1]
            @test face_digit(south_halo_west_value) == 6
            @test i_digits(south_halo_west_value) == 1
            @test j_digits(south_halo_west_value) == 32

            # Grid point [i, j] = [32, 0] in 1S halo should be from [i, j] = [32, 32] in 6N boundary.
            south_halo_east_value = get_face(field, 1)[32, 0, 1]
            @test face_digit(south_halo_east_value) == 6
            @test i_digits(south_halo_east_value) == 32
            @test j_digits(south_halo_east_value) == 32

            south_halo_values = south_halo(get_face(field, 1), include_corners=false)[:] |> Array
            @test all(face_digit.(south_halo_values) .== 6)
            @test all(i_digits.(south_halo_values) .== 1:32)
            @test all(j_digits.(south_halo_values) .== 32)
        end

        @testset "1N halo <- 3W boundary halo exchange" begin
            # Grid point [i, j] = [1, 33] in 1N halo should be from [i, j] = [1, 32] in 3W boundary.
            north_halo_west_value = get_face(field, 1)[1, 33, 1]
            @test face_digit(north_halo_west_value) == 3
            @test i_digits(north_halo_west_value) == 1
            @test j_digits(north_halo_west_value) == 32

            # Grid point [i, j] = [32, 33] in 1N halo should be from [i, j] = [1, 1] in 3W boundary.
            north_halo_east_value = get_face(field, 1)[32, 33, 1]
            @test face_digit(north_halo_east_value) == 3
            @test i_digits(north_halo_east_value) == 1
            @test j_digits(north_halo_east_value) == 1

            north_halo_values = north_halo(get_face(field, 1), include_corners=false)[:] |> Array
            @test all(face_digit.(north_halo_values) .== 3)
            @test all(i_digits.(north_halo_values) .== 1)
            @test all(j_digits.(north_halo_values) .== reverse(1:32))
        end

        @testset "2W halo <- 1E boundary halo exchange" begin
            # Grid point [i, j] = [0, 1] in 2W halo should be from [i, j] = [32, 1] in 1E boundary.
            west_halo_south_value = get_face(field, 2)[0, 1, 1]
            @test face_digit(west_halo_south_value) == 1
            @test i_digits(west_halo_south_value) == 32
            @test j_digits(west_halo_south_value) == 1

            # Grid point [i, j] = [0, 32] in 2W halo should be from [i, j] = [32, 32] in 1E boundary.
            west_halo_north_value = get_face(field, 2)[0, 32, 1]
            @test face_digit(west_halo_north_value) == 1
            @test i_digits(west_halo_north_value) == 32
            @test j_digits(west_halo_north_value) == 32

            west_halo_values = west_halo(get_face(field, 2), include_corners=false)[:] |> Array
            @test all(face_digit.(west_halo_values) .== 1)
            @test all(i_digits.(west_halo_values) .== 32)
            @test all(j_digits.(west_halo_values) .== 1:32)
        end

        @testset "2E halo <- 4S boundary halo exchange" begin
            # Grid point [i, j] = [33, 1] in 2E halo should be from [i, j] = [32, 1] in 4S boundary.
            east_halo_south_value = get_face(field, 2)[33, 1, 1]
            @test face_digit(east_halo_south_value) == 4
            @test i_digits(east_halo_south_value) == 32
            @test j_digits(east_halo_south_value) == 1

            # Grid point [i, j] = [33, 32] in 2E halo should be from [i, j] = [1, 1] in 4S boundary.
            east_halo_north_value = get_face(field, 2)[33, 32, 1]
            @test face_digit(east_halo_north_value) == 4
            @test i_digits(east_halo_north_value) == 1
            @test j_digits(east_halo_north_value) == 1

            east_halo_values = east_halo(get_face(field, 2), include_corners=false)[:] |> Array
            @test all(face_digit.(east_halo_values) .== 4)
            @test all(i_digits.(east_halo_values) .== reverse(1:32))
            @test all(j_digits.(east_halo_values) .== 1)
        end

        @testset "2S halo <- 6E boundary halo exchange" begin
            # Grid point [i, j] = [1, 0] in 2S halo should be from [i, j] = [32, 32] in 6E boundary.
            south_halo_west_value = get_face(field, 2)[1, 0, 1]
            @test face_digit(south_halo_west_value) == 6
            @test i_digits(south_halo_west_value) == 32
            @test j_digits(south_halo_west_value) == 32

            # Grid point [i, j] = [32, 0] in 2S halo should be from [i, j] = [32, 1] in 6E boundary.
            south_halo_east_value = get_face(field, 2)[32, 0, 1]
            @test face_digit(south_halo_east_value) == 6
            @test i_digits(south_halo_east_value) == 32
            @test j_digits(south_halo_east_value) == 1

            south_halo_values = south_halo(get_face(field, 2), include_corners=false)[:] |> Array
            @test all(face_digit.(south_halo_values) .== 6)
            @test all(i_digits.(south_halo_values) .== 32)
            @test all(j_digits.(south_halo_values) .== reverse(1:32))
        end

        @testset "2N halo <- 3S boundary halo exchange" begin
            # Grid point [i, j] = [1, 33] in 2N halo should be from [i, j] = [1, 1] in 3S boundary.
            north_halo_west_value = get_face(field, 2)[1, 33, 1]
            @test face_digit(north_halo_west_value) == 3
            @test i_digits(north_halo_west_value) == 1
            @test j_digits(north_halo_west_value) == 1

            # Grid point [i, j] = [32, 33] in 2N halo should be from [i, j] = [32, 1] in 3S boundary.
            north_halo_east_value = get_face(field, 2)[32, 33, 1]
            @test face_digit(north_halo_east_value) == 3
            @test i_digits(north_halo_east_value) == 32
            @test j_digits(north_halo_east_value) == 1

            north_halo_values = north_halo(get_face(field, 2), include_corners=false)[:] |> Array
            @test all(face_digit.(north_halo_values) .== 3)
            @test all(i_digits.(north_halo_values) .== 1:32)
            @test all(j_digits.(north_halo_values) .== 1)
        end

        @testset "3W halo <- 1N boundary halo exchange" begin
            # Grid point [i, j] = [0, 1] in 3W halo should be from [i, j] = [32, 32] in 1N boundary.
            west_halo_south_value = get_face(field, 3)[0, 1, 1]
            @test face_digit(west_halo_south_value) == 1
            @test i_digits(west_halo_south_value) == 32
            @test j_digits(west_halo_south_value) == 32

            # Grid point [i, j] = [0, 32] in 3W halo should be from [i, j] = [1, 32] in 1N boundary.
            west_halo_north_value = get_face(field, 3)[0, 32, 1]
            @test face_digit(west_halo_north_value) == 1
            @test i_digits(west_halo_north_value) == 1
            @test j_digits(west_halo_north_value) == 32

            west_halo_values = west_halo(get_face(field, 3), include_corners=false)[:] |> Array
            @test all(face_digit.(west_halo_values) .== 1)
            @test all(i_digits.(west_halo_values) .== reverse(1:32))
            @test all(j_digits.(west_halo_values) .== 32)
        end

        @testset "3E halo <- 4W boundary halo exchange" begin
            # Grid point [i, j] = [33, 1] in 3E halo should be from [i, j] = [1, 1] in 4W boundary.
            east_halo_south_value = get_face(field, 3)[33, 1, 1]
            @test face_digit(east_halo_south_value) == 4
            @test i_digits(east_halo_south_value) == 1
            @test j_digits(east_halo_south_value) == 1

            # Grid point [i, j] = [33, 32] in 3E halo should be from [i, j] = [1, 32] in 4W boundary.
            east_halo_north_value = get_face(field, 3)[33, 32, 1]
            @test face_digit(east_halo_north_value) == 4
            @test i_digits(east_halo_north_value) == 1
            @test j_digits(east_halo_north_value) == 32

            east_halo_values = east_halo(get_face(field, 3), include_corners=false)[:] |> Array
            @test all(face_digit.(east_halo_values) .== 4)
            @test all(i_digits.(east_halo_values) .== 1)
            @test all(j_digits.(east_halo_values) .== 1:32)
        end

        @testset "3S halo <- 2N boundary halo exchange" begin
            # Grid point [i, j] = [1, 0] in 3S halo should be from [i, j] = [1, 32] in 2N boundary.
            south_halo_west_value = get_face(field, 3)[1, 0, 1]
            @test face_digit(south_halo_west_value) == 2
            @test i_digits(south_halo_west_value) == 1
            @test j_digits(south_halo_west_value) == 32

            # Grid point [i, j] = [32, 0] in 3S halo should be from [i, j] = [32, 32] in 2N boundary.
            south_halo_east_value = get_face(field, 3)[32, 0, 1]
            @test face_digit(south_halo_east_value) == 2
            @test i_digits(south_halo_east_value) == 32
            @test j_digits(south_halo_east_value) == 32

            south_halo_values = south_halo(get_face(field, 3), include_corners=false)[:] |> Array
            @test all(face_digit.(south_halo_values) .== 2)
            @test all(i_digits.(south_halo_values) .== 1:32)
            @test all(j_digits.(south_halo_values) .== 32)
        end

        @testset "3N halo <- 5W boundary halo exchange" begin
            # Grid point [i, j] = [1, 33] in 3N halo should be from [i, j] = [1, 32] in 5W boundary.
            north_halo_west_value = get_face(field, 3)[1, 33, 1]
            @test face_digit(north_halo_west_value) == 5
            @test i_digits(north_halo_west_value) == 1
            @test j_digits(north_halo_west_value) == 32

            # Grid point [i, j] = [32, 33] in 3N halo should be from [i, j] = [1, 1] in 5W boundary.
            north_halo_east_value = get_face(field, 3)[32, 33, 1]
            @test face_digit(north_halo_east_value) == 5
            @test i_digits(north_halo_east_value) == 1
            @test j_digits(north_halo_east_value) == 1

            north_halo_values = north_halo(get_face(field, 3), include_corners=false)[:] |> Array
            @test all(face_digit.(north_halo_values) .== 5)
            @test all(i_digits.(north_halo_values) .== 1)
            @test all(j_digits.(north_halo_values) .== reverse(1:32))
        end

        @testset "4W halo <- 3E boundary halo exchange" begin
            # Grid point [i, j] = [0, 1] in 4W halo should be from [i, j] = [32, 1] in 3E boundary.
            west_halo_south_value = get_face(field, 4)[0, 1, 1]
            @test face_digit(west_halo_south_value) == 3
            @test i_digits(west_halo_south_value) == 32
            @test j_digits(west_halo_south_value) == 1

            # Grid point [i, j] = [0, 32] in 4W halo should be from [i, j] = [32, 32] in 1N boundary.
            west_halo_north_value = get_face(field, 4)[0, 32, 1]
            @test face_digit(west_halo_north_value) == 3
            @test i_digits(west_halo_north_value) == 32
            @test j_digits(west_halo_north_value) == 32

            west_halo_values = west_halo(get_face(field, 4), include_corners=false)[:] |> Array
            @test all(face_digit.(west_halo_values) .== 3)
            @test all(i_digits.(west_halo_values) .== 32)
            @test all(j_digits.(west_halo_values) .== 1:32)
        end

        @testset "4E halo <- 6S boundary halo exchange" begin
            # Grid point [i, j] = [33, 1] in 4E halo should be from [i, j] = [32, 1] in 6S boundary.
            east_halo_south_value = get_face(field, 4)[33, 1, 1]
            @test face_digit(east_halo_south_value) == 6
            @test i_digits(east_halo_south_value) == 32
            @test j_digits(east_halo_south_value) == 1

            # Grid point [i, j] = [33, 32] in 4E halo should be from [i, j] = [1, 1] in 6S boundary.
            east_halo_north_value = get_face(field, 4)[33, 32, 1]
            @test face_digit(east_halo_north_value) == 6
            @test i_digits(east_halo_north_value) == 1
            @test j_digits(east_halo_north_value) == 1

            east_halo_values = east_halo(get_face(field, 4), include_corners=false)[:] |> Array
            @test all(face_digit.(east_halo_values) .== 6)
            @test all(i_digits.(east_halo_values) .== reverse(1:32))
            @test all(j_digits.(east_halo_values) .== 1)
        end

        @testset "4S halo <- 2E boundary halo exchange" begin
            # Grid point [i, j] = [1, 0] in 4S halo should be from [i, j] = [32, 32] in 2E boundary.
            south_halo_west_value = get_face(field, 4)[1, 0, 1]
            @test face_digit(south_halo_west_value) == 2
            @test i_digits(south_halo_west_value) == 32
            @test j_digits(south_halo_west_value) == 32

            # Grid point [i, j] = [32, 0] in 4S halo should be from [i, j] = [32, 1] in 2E boundary.
            south_halo_east_value = get_face(field, 4)[32, 0, 1]
            @test face_digit(south_halo_east_value) == 2
            @test i_digits(south_halo_east_value) == 32
            @test j_digits(south_halo_east_value) == 1

            south_halo_values = south_halo(get_face(field, 4), include_corners=false)[:] |> Array
            @test all(face_digit.(south_halo_values) .== 2)
            @test all(i_digits.(south_halo_values) .== 32)
            @test all(j_digits.(south_halo_values) .== reverse(1:32))
        end

        @testset "4N halo <- 5S boundary halo exchange" begin
            # Grid point [i, j] = [1, 33] in 4N halo should be from [i, j] = [1, 1] in 5S boundary.
            north_halo_west_value = get_face(field, 4)[1, 33, 1]
            @test face_digit(north_halo_west_value) == 5
            @test i_digits(north_halo_west_value) == 1
            @test j_digits(north_halo_west_value) == 1

            # Grid point [i, j] = [32, 33] in 4N halo should be from [i, j] = [32, 1] in 5S boundary.
            north_halo_east_value = get_face(field, 4)[32, 33, 1]
            @test face_digit(north_halo_east_value) == 5
            @test i_digits(north_halo_east_value) == 32
            @test j_digits(north_halo_east_value) == 1

            north_halo_values = north_halo(get_face(field, 4), include_corners=false)[:] |> Array
            @test all(face_digit.(north_halo_values) .== 5)
            @test all(i_digits.(north_halo_values) .== 1:32)
            @test all(j_digits.(north_halo_values) .== 1)
        end

        ## TODO: Test faces 5 and 6 (or generalize the test to all faces)!

        CUDA.allowscalar(false)

    end
end

for arch in archs
    @testset "Cubed sphere velocity halo exchange [$(typeof(arch))]" begin
        @info "  Testing cubed sphere velocity halo exchange [$(typeof(arch))]..."

        grid = ConformalCubedSphereGrid(cs32_filepath, architecture=arch, Nz=1, z=(-1, 0))

        u_field = XFaceField(arch, grid)
        v_field = YFaceField(arch, grid)

        ## We will fill each grid point with a 6-digit integer "ufiijj" where
        ## the u digit is 1 for u and 2 for v, the f digit is the face number,
        ## the ii digits are the i index, and the jj digits are the j index.
        ## We then check that the halo exchange happened correctly (including
        ## any sign changes).

        U_DIGIT = 1
        V_DIGIT = 2

        uv_digit(n) = digits(abs(Int(n)))[6]
        face_digit(n) = digits(abs(Int(n)))[5]
        i_digits(n) = digits(abs(Int(n)))[3:4] |> undigits
        j_digits(n) = digits(abs(Int(n)))[1:2] |> undigits

        for (face_number, u_field_face) in enumerate(faces(u_field))
            for i in 1:u_field_face.grid.Nx+1, j in 1:u_field_face.grid.Ny
                CUDA.@allowscalar u_field_face[i, j, 1] = parse(Int, @sprintf("%d%d%02d%02d", U_DIGIT, face_number, i, j))
            end
        end

        for (face_number, v_field_face) in enumerate(faces(v_field))
            for i in 1:v_field_face.grid.Nx, j in 1:v_field_face.grid.Ny+1
                CUDA.@allowscalar v_field_face[i, j, 1] = parse(Int, @sprintf("%d%d%02d%02d", V_DIGIT, face_number, i, j))
            end
        end

        fill_horizontal_velocity_halos!(u_field, v_field)

        CUDA.allowscalar(true)

        # @testset "Source and destination faces are correct" begin
        #     for field in (u_field, v_field), (face_number, field_face) in enumerate(faces(field))
        #         west_halo_vals =  west_halo(field_face, include_corners=false)
        #         east_halo_vals =  east_halo(field_face, include_corners=false)
        #         south_halo_vals = south_halo(field_face, include_corners=false)
        #         north_halo_vals = north_halo(field_face, include_corners=false)

        #         @test all(face_digit.(west_halo_vals)  .== grid.face_connectivity[face_number].west.face)
        #         @test all(face_digit.(east_halo_vals)  .== grid.face_connectivity[face_number].east.face)
        #         @test all(face_digit.(south_halo_vals) .== grid.face_connectivity[face_number].south.face)
        #         @test all(face_digit.(north_halo_vals) .== grid.face_connectivity[face_number].north.face)
        #     end
        # end

        @testset "(1W halo <- 5N boundary) horizontal velocity halo exchange" begin
            # Grid point u[i, j] = u[0, 1] in 1W halo should be from +v[i, j] = +v[32, 32] in 5N boundary.
            u_west_halo_south_value = get_face(u_field, 1)[0, 1, 1]
            @test uv_digit(u_west_halo_south_value) == V_DIGIT
            @test sign(u_west_halo_south_value) == +1
            @test face_digit(u_west_halo_south_value) == 5
            @test i_digits(u_west_halo_south_value) == 32
            @test j_digits(u_west_halo_south_value) == 32

            # Grid point u[i, j] = u[0, 32] in 1W halo should be from +v[i, j] = +v[1, 32] in 5N boundary.
            u_west_halo_north_value = get_face(u_field, 1)[0, 32, 1]
            @test uv_digit(u_west_halo_north_value) == V_DIGIT
            @test sign(u_west_halo_north_value) == +1
            @test face_digit(u_west_halo_north_value) == 5
            @test i_digits(u_west_halo_north_value) == 1
            @test j_digits(u_west_halo_north_value) == 32

            u_west_halo_values = west_halo(get_face(u_field, 1), include_corners=false)[:] |> Array
            @test all(uv_digit.(u_west_halo_values) .== V_DIGIT)
            @test all(sign.(u_west_halo_values) .== +1)
            @test all(face_digit.(u_west_halo_values) .== 5)
            @test all(i_digits.(u_west_halo_values) .== reverse(1:32))
            @test all(j_digits.(u_west_halo_values) .== 32)

            # Note: The 1W v halo <- 5N -u boundary exchange involves a sign change and an index shift.

            # Grid point v[i, j] = v[0, 1] in 1W halo should be from -u[i, j] = -u[1, 32] in 6NE corner!
            v_west_halo_south_value = get_face(v_field, 1)[0, 1, 1]
            @test uv_digit(v_west_halo_south_value) == U_DIGIT
            @test sign(v_west_halo_south_value) == -1
            @test face_digit(v_west_halo_south_value) == 6
            @test i_digits(v_west_halo_south_value) == 1
            @test j_digits(v_west_halo_south_value) == 32

            # Grid point v[i, j] = u[0, 33] in 1W halo should be from -u[i, j] = -u[2, 32] in 5N boundary.
            v_west_halo_north_value = get_face(v_field, 1)[0, 32, 1]
            @test uv_digit(v_west_halo_north_value) == U_DIGIT
            @test sign(v_west_halo_north_value) == -1
            @test face_digit(v_west_halo_north_value) == 5
            @test i_digits(v_west_halo_north_value) == 2
            @test j_digits(v_west_halo_north_value) == 32

            v_west_halo_values = west_halo(get_face(v_field, 1), include_corners=false)[:] |> Array
            @test all(uv_digit.(v_west_halo_values) .== U_DIGIT)
            @test all(sign.(v_west_halo_values) .== -1)
            # @test all(face_digit.(v_west_halo_values) .== 5)
            # @test all(i_digits.(v_west_halo_values) .== reverse(2:33))
            @test all(j_digits.(v_west_halo_values) .== 32)
        end

        @testset "(1E halo <- 2W boundary) horizontal velocity halo exchange" begin
            # Grid point u[i, j] = u[33, 1] in 1E halo should be from +u[i, j] = +u[1, 1] in 2W boundary.
            u_east_halo_south_value = get_face(u_field, 1)[33, 1, 1]
            @test uv_digit(u_east_halo_south_value) == U_DIGIT
            @test sign(u_east_halo_south_value) == +1
            @test face_digit(u_east_halo_south_value) == 2
            @test i_digits(u_east_halo_south_value) == 1
            @test j_digits(u_east_halo_south_value) == 1

            # Grid point u[i, j] = u[33, 32] in 1E halo should be from +u[i, j] = +u[1, 32] in 2W boundary.
            u_east_halo_north_value = get_face(u_field, 1)[33, 32, 1]
            @test uv_digit(u_east_halo_north_value) == U_DIGIT
            @test sign(u_east_halo_north_value) == +1
            @test face_digit(u_east_halo_north_value) == 2
            @test i_digits(u_east_halo_north_value) == 1
            @test j_digits(u_east_halo_north_value) == 32

            u_east_halo_values = east_halo(get_face(u_field, 1), include_corners=false)[:] |> Array
            @test all(uv_digit.(u_east_halo_values) .== U_DIGIT)
            @test all(sign.(u_east_halo_values) .== +1)
            @test all(face_digit.(u_east_halo_values) .== 2)
            @test all(i_digits.(u_east_halo_values) .== 1)
            @test all(j_digits.(u_east_halo_values) .== 1:32)

            # Grid point v[i, j] = v[33, 1] in 1E halo should be from +v[i, j] = +v[1, 1] in 2W boundary.
            v_east_halo_south_value = get_face(v_field, 1)[33, 1, 1]
            @test uv_digit(v_east_halo_south_value) == V_DIGIT
            @test sign(v_east_halo_south_value) == +1
            @test face_digit(v_east_halo_south_value) == 2
            @test i_digits(v_east_halo_south_value) == 1
            @test j_digits(v_east_halo_south_value) == 1

            # # Grid point v[i, j] = v[33, 32] in 1E halo should be from +v[i, j] = +v[1, 32] in 2W boundary.
            v_east_halo_north_value = get_face(v_field, 1)[33, 32, 1]
            @test uv_digit(v_east_halo_north_value) == V_DIGIT
            @test sign(v_east_halo_north_value) == +1
            @test face_digit(v_east_halo_north_value) == 2
            @test i_digits(v_east_halo_north_value) == 1
            @test j_digits(v_east_halo_north_value) == 32

            v_east_halo_values = east_halo(get_face(v_field, 1), include_corners=false)[:] |> Array
            @test all(uv_digit.(v_east_halo_values) .== V_DIGIT)
            @test all(sign.(v_east_halo_values) .== +1)
            @test all(face_digit.(v_east_halo_values) .== 2)
            @test all(i_digits.(v_east_halo_values) .== 1)
            @test all(j_digits.(v_east_halo_values) .== 1:32)
        end

        @testset "(1S halo <- 6N boundary) horizontal velocity halo exchange" begin
            # Grid point u[i, j] = u[1, 0] in 1S halo should be from +u[i, j] = +u[1, 32] in 6N boundary.
            u_south_halo_west_value = get_face(u_field, 1)[1, 0, 1]
            @test uv_digit(u_south_halo_west_value) == U_DIGIT
            @test sign(u_south_halo_west_value) == +1
            @test face_digit(u_south_halo_west_value) == 6
            @test i_digits(u_south_halo_west_value) == 1
            @test j_digits(u_south_halo_west_value) == 32

            # Grid point u[i, j] = u[32, 0] in 1S halo should be from +u[i, j] = +u[32, 32] in 6N boundary.
            u_south_halo_east_value = get_face(u_field, 1)[32, 0, 1]
            @test uv_digit(u_south_halo_east_value) == U_DIGIT
            @test sign(u_south_halo_east_value) == +1
            @test face_digit(u_south_halo_east_value) == 6
            @test i_digits(u_south_halo_east_value) == 32
            @test j_digits(u_south_halo_east_value) == 32

            u_south_halo_values = south_halo(get_face(u_field, 1), include_corners=false)[:] |> Array
            @test all(uv_digit.(u_south_halo_values) .== U_DIGIT)
            @test all(sign.(u_south_halo_values) .== +1)
            @test all(face_digit.(u_south_halo_values) .== 6)
            @test all(i_digits.(u_south_halo_values) .== 1:32)
            @test all(j_digits.(u_south_halo_values) .== 32)

            # Grid point v[i, j] = v[1, 0] in 1S halo should be from +v[i, j] = +v[1, 32] in 6N boundary.
            v_south_halo_west_value = get_face(v_field, 1)[1, 0, 1]
            @test uv_digit(v_south_halo_west_value) == V_DIGIT
            @test sign(v_south_halo_west_value) == +1
            @test face_digit(v_south_halo_west_value) == 6
            @test i_digits(v_south_halo_west_value) == 1
            @test j_digits(v_south_halo_west_value) == 32

            # # Grid point v[i, j] = v[32, 0] in 1S halo should be from +v[i, j] = +v[32, 32] in 6N boundary.
            v_south_halo_east_value = get_face(v_field, 1)[32, 0, 1]
            @test uv_digit(v_south_halo_east_value) == V_DIGIT
            @test sign(v_south_halo_east_value) == +1
            @test face_digit(v_south_halo_east_value) == 6
            @test i_digits(v_south_halo_east_value) == 32
            @test j_digits(v_south_halo_east_value) == 32

            v_south_halo_values = south_halo(get_face(v_field, 1), include_corners=false)[:] |> Array
            @test all(uv_digit.(v_south_halo_values) .== V_DIGIT)
            @test all(sign.(v_south_halo_values) .== +1)
            @test all(face_digit.(v_south_halo_values) .== 6)
            @test all(i_digits.(v_south_halo_values) .== 1:32)
            @test all(j_digits.(v_south_halo_values) .== 32)
        end

        @testset "(1N halo <- 3W boundary) horizontal velocity halo exchange" begin

            # Note: The 1N u halo <- 3W -v boundary exchange involves a sign change and an index shift.

            # Grid point u[i, j] = u[1, 33] in 1N halo should be from -v[i, j] = -u[1, 32] in 5NE corner!
            u_nouth_halo_west_value = get_face(u_field, 1)[1, 33, 1]
            @test uv_digit(u_nouth_halo_west_value) == U_DIGIT
            @test sign(u_nouth_halo_west_value) == -1
            @test face_digit(u_nouth_halo_west_value) == 5
            @test i_digits(u_nouth_halo_west_value) == 1
            @test j_digits(u_nouth_halo_west_value) == 32

            # Grid point u[i, j] = u[32, 33] in 1N halo should be from -v[i, j] = -v[1, 2] in 3W boundary.
            u_north_halo_east_value = get_face(u_field, 1)[32, 33, 1]
            @test uv_digit(u_north_halo_east_value) == V_DIGIT
            @test sign(u_north_halo_east_value) == -1
            @test face_digit(u_north_halo_east_value) == 3
            @test i_digits(u_north_halo_east_value) == 1
            @test j_digits(u_north_halo_east_value) == 2

            u_north_halo_values = north_halo(get_face(u_field, 1), include_corners=false)[:] |> Array
            # @test all(uv_digit.(u_north_halo_values) .== V_DIGIT)
            @test all(sign.(u_north_halo_values) .== -1)
            # @test all(face_digit.(u_north_halo_values) .== 3)
            @test all(i_digits.(u_north_halo_values) .== 1)
            # @test all(j_digits.(u_north_halo_values) .== reverse(2:33))

            # Grid point v[i, j] = v[1, 33] in 1N halo should be from +u[i, j] = +u[1, 32] in 3W boundary.
            v_north_halo_west_value = get_face(v_field, 1)[1, 33, 1]
            @test uv_digit(v_north_halo_west_value) == U_DIGIT
            @test sign(v_north_halo_west_value) == +1
            @test face_digit(v_north_halo_west_value) == 3
            @test i_digits(v_north_halo_west_value) == 1
            @test j_digits(v_north_halo_west_value) == 32

            # # Grid point v[i, j] = v[32, 33] in 1N halo should be from +u[i, j] = +u[1, 1] in 3W boundary.
            v_north_halo_east_value = get_face(v_field, 1)[32, 33, 1]
            @test uv_digit(v_north_halo_east_value) == U_DIGIT
            @test sign(v_north_halo_east_value) == +1
            @test face_digit(v_north_halo_east_value) == 3
            @test i_digits(v_north_halo_east_value) == 1
            @test j_digits(v_north_halo_east_value) == 1

            v_north_halo_values = north_halo(get_face(v_field, 1), include_corners=false)[:] |> Array
            @test all(uv_digit.(v_north_halo_values) .== U_DIGIT)
            @test all(sign.(v_north_halo_values) .== +1)
            @test all(face_digit.(v_north_halo_values) .== 3)
            @test all(i_digits.(v_north_halo_values) .== 1)
            @test all(j_digits.(v_north_halo_values) .== reverse(1:32))
        end

        ## TODO: Test the other faces, especially face 2? I hope we can generalize this...

        @testset "Velocities at the corners (MITgcm regression)" begin
            # Face 1
            @test get_face(u_field, 1)[1,  33, 1] == -150132
            @test get_face(u_field, 1)[33, 0,  1] ==  220101

            @test get_face(v_field, 1)[0,   1, 1] == -160132
            @test get_face(v_field, 1)[0,  33, 1] == -150132
            @test get_face(v_field, 1)[33, 33, 1] ==  230101

            # Face 2
            @test get_face(u_field, 2)[1,  0,  1] == -213201
            @test get_face(u_field, 2)[33, 33, 1] ==  140101

            @test get_face(v_field, 2)[33,  1, 1] == -263201
            @test get_face(v_field, 2)[0,  33, 1] ==  130101
            @test get_face(v_field, 2)[33, 33, 1] == -140101

            # Face 3
            @test get_face(u_field, 3)[1,  33, 1] == -110132

            @test get_face(v_field, 3)[0,  33, 1] == -110132
            @test get_face(v_field, 3)[33, 33, 1] ==  250101

            # Face 4
            @test get_face(u_field, 4)[33, 33, 1] ==  160101

            @test get_face(v_field, 4)[33,  1, 1] == -223201
            @test get_face(v_field, 4)[0,  33, 1] ==  150101
            @test get_face(v_field, 4)[33, 33, 1] == -160101

            # Face 5
            @test get_face(v_field, 5)[0,  33, 1] == -130132
            @test get_face(v_field, 5)[33, 33, 1] ==  210101

            # Face 6
            @test get_face(v_field, 6)[0,  33, 1] ==  110101
            @test get_face(v_field, 6)[33, 33, 1] == -120101
        end

        CUDA.allowscalar(false)

    end
end
