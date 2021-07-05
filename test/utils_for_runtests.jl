using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, RungeKutta3TimeStepper, update_state!

#####
##### Useful kernels
#####

@kernel function ∇²!(∇²f, grid, f)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²f[i, j, k] = ∇²ᶜᶜᶜ(i, j, k, grid, f)
end

@kernel function divergence!(grid, u, v, w, div)
    i, j, k = @index(Global, NTuple)
    @inbounds div[i, j, k] = divᶜᶜᶜ(i, j, k, grid, u, v, w)
end

function compute_∇²!(∇²ϕ, ϕ, arch, grid)
    fill_halo_regions!(ϕ)
    child_arch = child_architecture(arch)

    event = launch!(child_arch, grid, :xyz, ∇²!, ∇²ϕ, grid, ϕ, dependencies=Event(device(child_arch)))

    wait(device(child_arch), event)

    fill_halo_regions!(∇²ϕ)

    return nothing
end

#####
##### Useful utilities
#####

const AB2Model = IncompressibleModel{<:QuasiAdamsBashforth2TimeStepper}
const RK3Model = IncompressibleModel{<:RungeKutta3TimeStepper}

# For time-stepping without a Simulation
function ab2_or_rk3_time_step!(model::AB2Model, Δt, n)
    n == 1 && update_state!(model)
    time_step!(model, Δt, euler=n==1)
    return nothing
end

function ab2_or_rk3_time_step!(model::RK3Model, Δt, n)
    n == 1 && update_state!(model)
    time_step!(model, Δt)
    return nothing
end

interior(a, grid) = view(a, grid.Hx+1:grid.Nx+grid.Hx,
                            grid.Hy+1:grid.Ny+grid.Hy,
                            grid.Hz+1:grid.Nz+grid.Hz)

datatuple(A) = NamedTuple{propertynames(A)}(Array(data(a)) for a in A)
datatuple(args, names) = NamedTuple{names}(a.data for a in args)

function get_model_field(field_name, model)
    if field_name ∈ (:u, :v, :w)
        return getfield(model.velocities, field_name)
    else
        return getfield(model.tracers, field_name)
    end
end

function get_output_tuple(output, iter, tuplename)
    file = jldopen(output.filepath, "r")
    output_tuple = file["timeseries/$tuplename/$iter"]
    close(file)
    return output_tuple
end

function run_script(replace_strings, script_name, script_filepath, module_suffix="")
    file_content = read(script_filepath, String)
    test_script_filepath = script_name * "_test.jl"

    for strs in replace_strings
        new_file_content = replace(file_content, strs[1] => strs[2])
        if new_file_content == file_content
            error("$(strs[1]) => $(strs[2]) replacement not found in $script_filepath. " *
                  "Make sure the script has not changed, otherwise the test might take a long time.")
            return false
        else
            file_content = new_file_content
        end
    end

    open(test_script_filepath, "w") do f
        # Wrap test script inside module to avoid polluting namespaces
        write(f, "module _Test_$script_name" * "_$module_suffix\n")
        write(f, file_content)
        write(f, "\nend # module")
    end

    try
        include(test_script_filepath)
    catch err
        @error sprint(showerror, err)

        # Print the content of the file to the test log, with line numbers, for debugging
        test_file_content = read(test_script_filepath, String)
        delineated_file_content = split(test_file_content, '\n')
        for (number, line) in enumerate(delineated_file_content)
            @printf("% 3d %s\n", number, line)
        end

        rm(test_script_filepath)
        return false
    end

    # Delete the test script (if it hasn't been deleted already)
    rm(test_script_filepath)

    return true
end

#####
##### Boundary condition utils
#####

discrete_func(i, j, grid, clock, model_fields) = - model_fields.u[i, j, grid.Nz]
parameterized_discrete_func(i, j, grid, clock, model_fields, p) = - p.μ * model_fields.u[i, j, grid.Nz]

parameterized_fun(ξ, η, t, p) = p.μ * cos(p.ω * t)
field_dependent_fun(ξ, η, t, u, v, w) = - w * sqrt(u^2 + v^2)
exploding_fun(ξ, η, t, T, S, p) = - p.μ * cosh(S - p.S0) * exp((T - p.T0) / p.λ)

# Many bc. Very many
                 integer_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, 1)
                   float_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, FT(π))
              irrational_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, π)
                   array_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, ArrayType(rand(FT, 1, 1)))
         simple_function_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, (ξ, η, t) -> exp(ξ) * cos(η) * sin(t))
  parameterized_function_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, parameterized_fun, parameters=(μ=0.1, ω=2π))
field_dependent_function_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, field_dependent_fun, field_dependencies=(:u, :v, :w))
       discrete_function_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, discrete_func, discrete_form=true)

       parameterized_discrete_function_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, parameterized_discrete_func, discrete_form=true, parameters=(μ=0.1,))
parameterized_field_dependent_function_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, exploding_fun, field_dependencies=(:T, :S), parameters=(S0=35, T0=100, μ=2π, λ=FT(2)))
