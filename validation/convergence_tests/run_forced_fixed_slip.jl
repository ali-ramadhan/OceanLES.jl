pushfirst!(LOAD_PATH, joinpath("..", ".."))

using CUDA
using Oceananigans

using ConvergenceTests.ForcedFlowFixedSlip: setup_and_run_xy, setup_and_run_xz

arch = CUDA.has_cuda() ? GPU() : CPU()

# Run 4 simulations:
Nx = [16, 32, 64, 128]
stop_time = 0.01
h = π / maximum(Nx)

Δt = 0.01 .* h^2 # Initial guess

# Modify so simulations stop on a dime
stop_iteration = round(Int, stop_time / Δt)
Δt = stop_time / stop_iteration

# Run (x, y) and (x, z) simulations
for N in Nx
    setup_and_run_xy(architecture=arch, Nx=N, Δt=Δt, stop_iteration=stop_iteration)
    setup_and_run_xz(architecture=arch, Nx=N, Δt=Δt, stop_iteration=stop_iteration)
end
