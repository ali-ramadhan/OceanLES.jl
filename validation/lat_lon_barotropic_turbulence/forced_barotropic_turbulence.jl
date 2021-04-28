# # Freely decaying barotropic turbulence on a latitude-longitude strip

using Oceananigans
using Oceananigans.Grids

using Oceananigans.Fields: FunctionField

using Oceananigans.Coriolis:
    HydrostaticSphericalCoriolis,
    VectorInvariantEnergyConserving,
    VectorInvariantEnstrophyConserving

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    HydrostaticFreeSurfaceModel,
    VerticalVorticityField,
    VectorInvariant,
    ExplicitFreeSurface,
    ImplicitFreeSurface

using Oceananigans.TurbulenceClosures:
    HorizontallyCurvilinearAnisotropicDiffusivity,
    HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity

using Oceananigans.Utils: prettytime, hours, day, days, years, year
using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval, IterationInterval

using Statistics
using JLD2
using Printf

using Oceananigans.AbstractOperations: AbstractGridMetric, _unary_operation

latitude = (-80, 80)
Δφ = latitude[2] - latitude[1]

resolution = 1/2 # degree
Nx = round(Int, 360 / resolution)
Ny = round(Int, Δφ / resolution)

# A spherical domain
@show grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                          longitude = (-180, 180),
                                          latitude = latitude,
                                          z = (-100, 0))

free_surface = ExplicitFreeSurface(gravitational_acceleration=0.2)

coriolis = HydrostaticSphericalCoriolis(scheme = VectorInvariantEnstrophyConserving())

equator_Δx = grid.radius * deg2rad(grid.Δλ)
polar_Δx = grid.radius * cosd(maximum(abs, grid.φᵃᶜᵃ)) * deg2rad(grid.Δλ)
diffusive_time_scale = 60days

@show const νh₂₀ = equator_Δx^2 / diffusive_time_scale
@show const νh₄₀ = 2e-5 * equator_Δx^4 / diffusive_time_scale
@inline νh₂(λ, φ, z, t) = νh₀ * cos(π * φ / 180)
@inline νh₄(λ, φ, z, t) = νh₄₀ * cos(π * φ / 180)

variable_horizontal_diffusivity = HorizontallyCurvilinearAnisotropicDiffusivity(νh=νh₂)
variable_horizontal_biharmonic_diffusivity = HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity(νh=νh₄)

# Set up forcing
using CUDA
using Oceananigans.Architectures: architecture
using Oceananigans.Operators: ∂yᶠᶜᵃ, ∂xᶜᶠᵃ
using Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.Fields: compute!

forcing_timescale = 1days
forcing_amplitude = polar_Δx^2 / forcing_timescale^2
drag_timescale = 2years

struct RandomOperand
    amplitude :: Float64
end

const RandomComputedField = ComputedField{X, Y, Z, S, <:RandomOperand} where {X, Y, Z, S}

function compute!(ψ::RandomComputedField)
    arch = architecture(ψ)
    ψ .= 0
    ψ .= ψ.operand.amplitude * CUDA.rand(size(ψ)...)
    fill_halo_regions!(ψ, arch)
    return nothing
end

# Build random streamfunction
arch = GPU()
dψdt = ComputedField(Face, Face, Center, RandomOperand(forcing_amplitude), arch, grid)

# Forcing by random streamfunction and bottom drag
@inline u_forcing_func(i, j, k, grid, clock, fields, μ) = @inbounds + ∂yᶠᶜᵃ(i, j, k, grid, fields.dψdt) - μ * fields.u[i, j, k]
@inline v_forcing_func(i, j, k, grid, clock, fields, μ) = @inbounds - ∂xᶜᶠᵃ(i, j, k, grid, fields.dψdt) - μ * fields.v[i, j, k]

u_forcing = Forcing(u_forcing_func; discrete_form=true, parameters=1/drag_timescale)
v_forcing = Forcing(v_forcing_func; discrete_form=true, parameters=1/drag_timescale)

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    architecture = arch,
                                    momentum_advection = VectorInvariant(),
                                    free_surface = free_surface,
                                    coriolis = nothing, # coriolis,
                                    tracers = nothing,
                                    buoyancy = nothing,
                                    auxiliary_fields = (dψdt=dψdt,),
                                    forcing = (u=u_forcing, v=v_forcing),
                                    closure = variable_horizontal_biharmonic_diffusivity)
                                    #closure = variable_horizontal_diffusivity)

g = model.free_surface.gravitational_acceleration

gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed

# Time-scale for gravity wave propagation across the smallest grid cell
wave_propagation_time_scale = min(grid.radius * cosd(maximum(abs, grid.φᵃᶜᵃ)) * deg2rad(grid.Δλ),
                                  grid.radius * deg2rad(grid.Δφ)) / gravity_wave_speed

set!(model,
     u = (x, y, z) -> sqrt(abs(sin(π * y / 180))) * rand(),
     v = (x, y, z) -> sqrt(abs(sin(π * y / 180))) * rand())

# Zero out mean motion
model.velocities.u .-= mean(model.velocities.u)
model.velocities.v .-= mean(model.velocities.v)

# Set target velocity to fraction of free surface velocity
max_u = maximum(model.velocities.u)
max_v = maximum(model.velocities.v)
max_speed = sqrt(max_u^2 + max_v^2)

target_speed = 0.5 * gravity_wave_speed
model.velocities.u ./= target_speed / max_speed
model.velocities.v ./= target_speed / max_speed

mutable struct Progress; interval_start_time::Float64; end

function (p::Progress)(sim)
    wall_time = (time_ns() - p.interval_start_time) * 1e-9

    @info @sprintf("Time: %s, iteration: %d, max(u): %.2e m s⁻¹, wall time: %s",
                   prettytime(sim.model.clock.time),
                   sim.model.clock.iteration,
                   maximum(sim.model.velocities.u),
                   prettytime(wall_time))

    p.interval_start_time = time_ns()

    return nothing
end

ζ = VerticalVorticityField(model)
compute!(ζ)

Δt = 0.2wave_propagation_time_scale

# Max Rossby number: $(maximum(abs, Ro))

@info """
    Maximum vertical vorticity: $(maximum(ζ))
    Inverse maximum vertical vorticity: $(prettytime(1/maximum(ζ)))
    Minimum wave propagation time scale: $(prettytime(wave_propagation_time_scale))
    Time step: $(prettytime(Δt))
"""

simulation = Simulation(model,
                        Δt = Δt,
                        stop_time = 10year,
                        iteration_interval = 100,
                        progress = Progress(time_ns()))

output_fields = merge(model.velocities, (η=model.free_surface.η, ζ=ζ))

output_prefix = "forced_barotropic_turbulence_Nx$(grid.Nx)_Ny$(grid.Ny)"

simulation.output_writers[:fields] = JLD2OutputWriter(model, (ζ = ζ,),
                                                      schedule = TimeInterval(60day),
                                                      prefix = output_prefix,
                                                      force = true)

run!(simulation)
