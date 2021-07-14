using Plots
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, TKEBasedVerticalDiffusivity

grid = RegularRectilinearGrid(size=8, z=(-64, 0), topology=(Flat, Flat, Bounded))

closure = TKEBasedVerticalDiffusivity(time_discretization=VerticallyImplicitTimeDiscretization())
                                      
Qᵇ = 0.0
Qᵘ = - 1e-4
Qᵛ = 0.0

u★ = (Qᵘ^2 + Qᵛ^2)^(1/4)
w★³ = Qᵇ * grid.Δz

Qᵉ = - closure.dissipation_parameter * (closure.surface_model.CᵂwΔ * w★³ + closure.surface_model.Cᵂu★ * u★^3)

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))
v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵛ))
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    tracers = (:b, :e),
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = FPlane(f=1e-4),
                                    boundary_conditions = (b=b_bcs, u=u_bcs, v=v_bcs),
                                    closure = closure)
                                    
N² = 1e-5
bᵢ(x, y, z) = N² * z
set!(model, b = bᵢ)

z = znodes(model.tracers.b)

u = view(interior(model.velocities.u), 1, 1, :)
v = view(interior(model.velocities.v), 1, 1, :)

b = view(interior(model.tracers.b), 1, 1, :)
e = view(interior(model.tracers.e), 1, 1, :)

Ku = view(interior(model.diffusivities.Kᵘ), 1, 1, :)
Kc = view(interior(model.diffusivities.Kᶜ), 1, 1, :)
Ke = view(interior(model.diffusivities.Kᵉ), 1, 1, :)

u_plot = plot(u, z, linewidth = 2, label = "u, t = 0", xlabel = "Velocities", ylabel = "z", legend=:bottomright)
plot!(u_plot, v, z, linewidth = 2, linestyle=:dash, label = "v, t = 0")

b_plot = plot(b, z, linewidth = 2, label = "t = 0", xlabel = "Buoyancy", ylabel = "z", legend=:bottomright)
e_plot = plot(e, z, linewidth = 2, label = "t = 0", xlabel = "TKE", ylabel = "z", legend=:bottomright)
              
simulation = Simulation(model, Δt = 10.0, stop_time=12hour)

run!(simulation)

plot!(u_plot, u, z, linewidth = 2, label = @sprintf("u, t = %s", prettytime(model.clock.time)))
plot!(u_plot, v, z, linewidth = 2, linestyle=:dash, label = @sprintf("v, t = %s", prettytime(model.clock.time)))

plot!(b_plot, b, z, linewidth = 2, label = @sprintf("t = %s", prettytime(model.clock.time)))
plot!(e_plot, e, z, linewidth = 2, label = @sprintf("t = %s", prettytime(model.clock.time)))

K_plot = plot(Kc, z, linewidth = 2, linestyle=:dash, label = @sprintf("Kᶜ, t = %s", prettytime(model.clock.time)), legend=:bottomright, xlabel="Diffusivities")
plot!(K_plot, Ke, z, linewidth = 3, alpha=0.6, label = @sprintf("Kᵉ, t = %s", prettytime(model.clock.time)))
plot!(K_plot, Ku, z, linewidth = 2, linestyle=:dot, label = @sprintf("Kᵘ, t = %s", prettytime(model.clock.time)))

eb_plot = plot(u_plot, b_plot, e_plot, K_plot, layout=(1, 4), size=(1200, 600))

display(eb_plot)

