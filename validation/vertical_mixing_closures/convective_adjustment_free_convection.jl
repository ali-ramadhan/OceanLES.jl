using Plots
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity

grid = RegularRectilinearGrid(size=32, z=(-64, 0), topology=(Flat, Flat, Bounded))

closure = ConvectiveAdjustmentVerticalDiffusivity(convective_κz=1, background_κz=1e-5)
                                      
Qᵇ = 1e-8
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    boundary_conditions = (b=b_bcs,),
                                    closure = closure)
                                    
N² = 1e-5
bᵢ(x, y, z) = N² * z
set!(model, b = bᵢ)

z = znodes(model.tracers.b)

b = view(interior(model.tracers.b), 1, 1, :)

b_plot = plot(b, z, linewidth = 2, label = "t = 0", xlabel = "Buoyancy", ylabel = "z", legend=:bottomright)
              
simulation = Simulation(model, Δt = 20.0, stop_time = 48hours)

run!(simulation)

plot!(b_plot, b, z, linewidth = 2, label = @sprintf("t = %s", prettytime(model.clock.time)))

display(b_plot)
