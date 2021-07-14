pushfirst!(LOAD_PATH, joinpath("..", ".."))

using CUDA
using Oceananigans

using ConvergenceTests.DoublyPeriodicTaylorGreen: setup_and_run

arch = CUDA.has_cuda() ? GPU() : CPU()
Nx = [8, 16, 32, 64, 128]
stop_time = 0.25

# Calculate time step
max_h = 2π / maximum(Nx)
Δt = 0.01 * max_h^2 # satisfy diffusive constraint for finest resolution
Nt = round(Int, stop_time / Δt)
Δt = stop_time / Nt

for N in Nx
    setup_and_run(architecture=arch, Nx=N, Δt=Δt, stop_iteration=Nt)
end
