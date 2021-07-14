using Plots
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, TKEBasedVerticalDiffusivity
using Oceananigans.TurbulenceClosures: RiDependentDiffusivityScaling

grid = RegularRectilinearGrid(size=32, z=(-64, 0), topology=(Flat, Flat, Bounded))

closure = TKEBasedVerticalDiffusivity(time_discretization=VerticallyImplicitTimeDiscretization())
                                      
Qᵇ = 1e-8
Qᵘ = 0.0
Qᵛ = 0.0

u★ = (Qᵘ^2 + Qᵛ^2)^(1/4)
w★ = Qᵇ * grid.Δz

Qᵉ = - closure.dissipation_parameter * (closure.surface_model.CᵂwΔ * w★ + closure.surface_model.Cᵂu★ * u★)

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))
v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵛ))
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    tracers = (:b, :e),
                                    buoyancy = BuoyancyTracer(),
                                    boundary_conditions = (; b=b_bcs),
                                    closure = closure)
                                    
N² = 1e-5
bᵢ(x, y, z) = N² * z
set!(model, b = bᵢ)

z = znodes(model.tracers.b)

b = view(interior(model.tracers.b), 1, 1, :)
e = view(interior(model.tracers.e), 1, 1, :)
Kc = view(interior(model.diffusivities.Kᶜ), 1, 1, :)
Ke = view(interior(model.diffusivities.Kᵉ), 1, 1, :)

b_plot = plot(b, z, linewidth = 2, label = "t = 0", xlabel = "Buoyancy", ylabel = "z", legend=:bottomright)
e_plot = plot(e, z, linewidth = 2, label = "t = 0", xlabel = "TKE", ylabel = "z", legend=:bottomright)
              
simulation = Simulation(model, Δt = 20.0, stop_time = 48hours)

run!(simulation)

plot!(b_plot, b, z, linewidth = 2, label = @sprintf("t = %s", prettytime(model.clock.time)))
plot!(e_plot, e, z, linewidth = 2, label = @sprintf("t = %s", prettytime(model.clock.time)))
K_plot = plot(Kc, z, linewidth = 2, linestyle=:dash, label = @sprintf("Kᶜ, t = %s", prettytime(model.clock.time)), legend=:bottomright, xlabel="Diffusivities")
plot!(K_plot, Ke, z, linewidth = 3, alpha=0.6, label = @sprintf("Kᵉ, t = %s", prettytime(model.clock.time)))

eb_plot = plot(b_plot, e_plot, K_plot, layout=(1, 3), size=(1200, 600))

display(eb_plot)
