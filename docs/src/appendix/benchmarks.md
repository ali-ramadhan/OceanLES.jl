# [Performance benchmarks](@id performance_benchmarks)

The performance benchmarking scripts in the
[`benchmarks`](https://github.com/CliMA/Oceananigans.jl/tree/master/benchmark)
directory of the git repository can be run to benchmark Oceananigans.jl on your machine.
They use [TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl) to nicely
format the benchmark results.

## Static ocean

This is a benchmark of a simple "static ocean" configuration. The time stepping and Poisson
solver still takes the same amount of time whether the ocean is static or active, so it is
indicative of actual performance. It tests the performance of a bare-bones
horizontally-periodic model with `topology = (Periodic, Periodic, Bounded)`.

```
Oceananigans v0.34.0 (DEVELOPMENT BRANCH)
Julia Version 1.4.2
Commit 44fa15b150* (2020-05-23 18:35 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) Silver 4214 CPU @ 2.20GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-8.0.1 (ORCJIT, skylake)
  GPU: TITAN V

 ──────────────────────────────────────────────────────────────────────────────────────
        Static ocean benchmarks                Time                   Allocations      
                                       ──────────────────────   ───────────────────────
           Tot / % measured:                 291s / 29.6%           27.7GiB / 0.50%    

 Section                       ncalls     time   %tot     avg     alloc   %tot      avg
 ──────────────────────────────────────────────────────────────────────────────────────
  16× 16× 16  [CPU, Float32]       10   15.6ms  0.02%  1.56ms   2.61MiB  1.84%   267KiB
  16× 16× 16  [CPU, Float64]       10   16.9ms  0.02%  1.69ms   2.61MiB  1.84%   267KiB
  16× 16× 16  [GPU, Float32]       10   53.4ms  0.06%  5.34ms   11.5MiB  8.14%  1.15MiB
  16× 16× 16  [GPU, Float64]       10   69.7ms  0.08%  6.97ms   11.5MiB  8.14%  1.15MiB
  32× 32× 32  [CPU, Float32]       10   54.6ms  0.06%  5.46ms   2.61MiB  1.84%   267KiB
  32× 32× 32  [CPU, Float64]       10   57.1ms  0.07%  5.71ms   2.61MiB  1.84%   267KiB
  32× 32× 32  [GPU, Float32]       10   57.5ms  0.07%  5.75ms   11.6MiB  8.15%  1.16MiB
  32× 32× 32  [GPU, Float64]       10   75.0ms  0.09%  7.50ms   11.6MiB  8.16%  1.16MiB
  64× 64× 64  [CPU, Float32]       10    424ms  0.49%  42.4ms   2.61MiB  1.84%   267KiB
  64× 64× 64  [CPU, Float64]       10    425ms  0.49%  42.5ms   2.61MiB  1.84%   267KiB
  64× 64× 64  [GPU, Float32]       10   61.7ms  0.07%  6.17ms   11.6MiB  8.16%  1.16MiB
  64× 64× 64  [GPU, Float64]       10   82.4ms  0.10%  8.24ms   11.6MiB  8.17%  1.16MiB
 128×128×128  [CPU, Float32]       10    3.67s  4.26%   367ms   2.61MiB  1.84%   267KiB
 128×128×128  [CPU, Float64]       10    3.64s  4.23%   364ms   2.61MiB  1.84%   267KiB
 128×128×128  [GPU, Float32]       10   74.8ms  0.09%  7.48ms   11.6MiB  8.16%  1.16MiB
 128×128×128  [GPU, Float64]       10   94.0ms  0.11%  9.40ms   11.6MiB  8.17%  1.16MiB
 256×256×256  [CPU, Float32]       10    38.5s  44.8%   3.85s   2.61MiB  1.84%   267KiB
 256×256×256  [CPU, Float64]       10    37.9s  44.1%   3.79s   2.61MiB  1.84%   267KiB
 256×256×256  [GPU, Float32]       10    350ms  0.41%  35.0ms   11.6MiB  8.18%  1.16MiB
 256×256×256  [GPU, Float64]       10    352ms  0.41%  35.2ms   11.6MiB  8.17%  1.16MiB
 ──────────────────────────────────────────────────────────────────────────────────────

CPU Float64 -> Float32 speedup:
 16× 16× 16 : 1.084
 32× 32× 32 : 1.046
 64× 64× 64 : 1.000
128×128×128 : 0.993
256×256×256 : 0.986

GPU Float64 -> Float32 speedup:
 16× 16× 16 : 1.304
 32× 32× 32 : 1.303
 64× 64× 64 : 1.335
128×128×128 : 1.257
256×256×256 : 1.004

CPU -> GPU speedup:
 16× 16× 16  [Float32]: 0.291
 16× 16× 16  [Float64]: 0.242
 32× 32× 32  [Float32]: 0.949
 32× 32× 32  [Float64]: 0.762
 64× 64× 64  [Float32]: 6.876
 64× 64× 64  [Float64]: 5.152
128×128×128  [Float32]: 49.036
128×128×128  [Float64]: 38.730
256×256×256  [Float32]: 109.868
256×256×256  [Float64]: 107.863
```

## Channel

This benchmark tests the channel model (`topology = (Periodic, Bounded, Bounded)`)
configuration which can be slower due to the use of a more complicated algorithm
(involving 2D cosine transforms) for the pressure solver in the current version
of Oceananigans.

```
Oceananigans v0.34.0 (DEVELOPMENT BRANCH)
Julia Version 1.4.2
Commit 44fa15b150* (2020-05-23 18:35 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) Silver 4214 CPU @ 2.20GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-8.0.1 (ORCJIT, skylake)
  GPU: TITAN V

 ──────────────────────────────────────────────────────────────────────────────────────
           Channel benchmarks                  Time                   Allocations      
                                       ──────────────────────   ───────────────────────
           Tot / % measured:                 453s / 19.5%           26.3GiB / 0.48%    

 Section                       ncalls     time   %tot     avg     alloc   %tot      avg
 ──────────────────────────────────────────────────────────────────────────────────────
  32× 32× 32  [CPU, Float32]       10   58.5ms  0.07%  5.85ms   2.84MiB  2.22%   291KiB
  32× 32× 32  [CPU, Float64]       10   60.8ms  0.07%  6.08ms   2.85MiB  2.22%   291KiB
  32× 32× 32  [GPU, Float32]       10   68.7ms  0.08%  6.87ms   12.6MiB  9.85%  1.26MiB
  32× 32× 32  [GPU, Float64]       10   88.2ms  0.10%  8.82ms   12.6MiB  9.85%  1.26MiB
  64× 64× 64  [CPU, Float32]       10    459ms  0.52%  45.9ms   2.84MiB  2.22%   291KiB
  64× 64× 64  [CPU, Float64]       10    442ms  0.50%  44.2ms   2.85MiB  2.22%   291KiB
  64× 64× 64  [GPU, Float32]       10   91.0ms  0.10%  9.10ms   12.8MiB  10.0%  1.28MiB
  64× 64× 64  [GPU, Float64]       10    108ms  0.12%  10.8ms   12.8MiB  10.0%  1.28MiB
 128×128×128  [CPU, Float32]       10    3.87s  4.38%   387ms   2.84MiB  2.22%   291KiB
 128×128×128  [CPU, Float64]       10    3.92s  4.44%   392ms   2.85MiB  2.22%   291KiB
 128×128×128  [GPU, Float32]       10    145ms  0.16%  14.5ms   13.2MiB  10.3%  1.32MiB
 128×128×128  [GPU, Float64]       10    163ms  0.18%  16.3ms   13.2MiB  10.3%  1.32MiB
 256×256×256  [CPU, Float32]       10    38.6s  43.6%   3.86s   2.85MiB  2.22%   292KiB
 256×256×256  [CPU, Float64]       10    38.7s  43.8%   3.87s   2.85MiB  2.22%   292KiB
 256×256×256  [GPU, Float32]       10    805ms  0.91%  80.5ms   14.0MiB  10.9%  1.40MiB
 256×256×256  [GPU, Float64]       10    805ms  0.91%  80.5ms   14.0MiB  10.9%  1.40MiB
 ──────────────────────────────────────────────────────────────────────────────────────

CPU Float64 -> Float32 speedup:
 32× 32× 32 : 1.040
 64× 64× 64 : 0.963
128×128×128 : 1.015
256×256×256 : 1.004

GPU Float64 -> Float32 speedup:
 32× 32× 32 : 1.283
 64× 64× 64 : 1.188
128×128×128 : 1.120
256×256×256 : 0.999

CPU -> GPU speedup:
 32× 32× 32  [Float32]: 0.851
 32× 32× 32  [Float64]: 0.689
 64× 64× 64  [Float32]: 5.044
 64× 64× 64  [Float64]: 4.088
128×128×128  [Float32]: 26.602
128×128×128  [Float64]: 24.097
256×256×256  [Float32]: 47.891
256×256×256  [Float64]: 48.116
```

## Tracers

This benchmark tests the performance impacts of running with various amounts of active
and passive tracers.

```
Oceananigans v0.34.0 (DEVELOPMENT BRANCH)
Julia Version 1.4.2
Commit 44fa15b150* (2020-05-23 18:35 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) Silver 4214 CPU @ 2.20GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-8.0.1 (ORCJIT, skylake)
  GPU: TITAN V

 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
                      Tracer benchmarks                             Time                   Allocations      
                                                            ──────────────────────   ───────────────────────
                      Tot / % measured:                           168s / 1.29%           18.6GiB / 0.77%    

 Section                                            ncalls     time   %tot     avg     alloc   %tot      avg
 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
  32× 32× 32 0 active +  0 passive [CPU, Float64]       10   42.4ms  1.96%  4.24ms   1.89MiB  1.29%   194KiB
  32× 32× 32 0 active +  1 passive [CPU, Float64]       10   48.3ms  2.23%  4.83ms   2.17MiB  1.49%   223KiB
  32× 32× 32 0 active +  2 passive [CPU, Float64]       10   53.5ms  2.46%  5.35ms   2.61MiB  1.78%   267KiB
  32× 32× 32 1 active +  0 passive [CPU, Float64]       10   48.9ms  2.25%  4.89ms   2.17MiB  1.49%   223KiB
  32× 32× 32 2 active +  0 passive [CPU, Float64]       10   56.0ms  2.58%  5.60ms   2.61MiB  1.78%   267KiB
  32× 32× 32 2 active +  3 passive [CPU, Float64]       10   77.0ms  3.55%  7.70ms   3.73MiB  2.55%   382KiB
  32× 32× 32 2 active +  5 passive [CPU, Float64]       10   90.4ms  4.16%  9.04ms   4.46MiB  3.05%   457KiB
  32× 32× 32 2 active + 10 passive [CPU, Float64]       10    127ms  5.86%  12.7ms   6.36MiB  4.35%   651KiB
 256×256×128 0 active +  0 passive [GPU, Float64]       10    146ms  6.75%  14.6ms   8.48MiB  5.80%   868KiB
 256×256×128 0 active +  1 passive [GPU, Float64]       10    162ms  7.45%  16.2ms   9.92MiB  6.78%  0.99MiB
 256×256×128 0 active +  2 passive [GPU, Float64]       10    174ms  8.03%  17.4ms   11.6MiB  7.92%  1.16MiB
 256×256×128 1 active +  0 passive [GPU, Float64]       10    160ms  7.39%  16.0ms   9.90MiB  6.77%  0.99MiB
 256×256×128 2 active +  0 passive [GPU, Float64]       10    177ms  8.18%  17.7ms   11.6MiB  7.93%  1.16MiB
 256×256×128 2 active +  3 passive [GPU, Float64]       10    222ms  10.2%  22.2ms   17.0MiB  11.6%  1.70MiB
 256×256×128 2 active +  5 passive [GPU, Float64]       10    255ms  11.8%  25.5ms   20.6MiB  14.1%  2.06MiB
 256×256×128 2 active + 10 passive [GPU, Float64]       10    328ms  15.1%  32.8ms   31.3MiB  21.4%  3.13MiB
 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
```

## Turbulence closures

This benchmark tests the performance impacts of various turbulent diffusivity closures
and large eddy simulation (LES) models.

```
Oceananigans v0.34.0 (DEVELOPMENT BRANCH)
Julia Version 1.4.2
Commit 44fa15b150* (2020-05-23 18:35 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) Silver 4214 CPU @ 2.20GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-8.0.1 (ORCJIT, skylake)
  GPU: TITAN V

 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
                         Turbulence closure benchmarks                                Time                   Allocations      
                                                                              ──────────────────────   ───────────────────────
                               Tot / % measured:                                    257s / 34.3%           25.0GiB / 0.47%    

 Section                                                              ncalls     time   %tot     avg     alloc   %tot      avg
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  32× 32× 32 AnisotropicDiffusivity [CPU, Float64]                        10   52.0ms  0.06%  5.20ms   2.62MiB  2.16%   268KiB
  32× 32× 32 AnisotropicDiffusivity [GPU, Float64]                        10   52.3ms  0.06%  5.23ms   11.6MiB  9.60%  1.16MiB
  32× 32× 32 IsotropicDiffusivity [CPU, Float64]                          10   52.7ms  0.06%  5.27ms   2.61MiB  2.15%   267KiB
  32× 32× 32 IsotropicDiffusivity [GPU, Float64]                          10   49.7ms  0.06%  4.97ms   11.6MiB  9.55%  1.16MiB
  32× 32× 32 SmagorinskyLilly [CPU, Float64]                              10   88.3ms  0.10%  8.83ms   2.73MiB  2.25%   280KiB
  32× 32× 32 SmagorinskyLilly [GPU, Float64]                              10   57.1ms  0.06%  5.71ms   12.5MiB  10.3%  1.25MiB
  32× 32× 32 VerstappenAnisotropicMinimumDissipation [CPU, Float64]       10   85.8ms  0.10%  8.58ms   2.93MiB  2.42%   300KiB
  32× 32× 32 VerstappenAnisotropicMinimumDissipation [GPU, Float64]       10   67.6ms  0.08%  6.76ms   14.0MiB  11.5%  1.40MiB
 256×256×128 AnisotropicDiffusivity [CPU, Float64]                        10    16.7s  19.0%   1.67s   2.62MiB  2.16%   268KiB
 256×256×128 AnisotropicDiffusivity [GPU, Float64]                        10    180ms  0.21%  18.0ms   11.7MiB  9.62%  1.17MiB
 256×256×128 IsotropicDiffusivity [CPU, Float64]                          10    16.5s  18.7%   1.65s   2.61MiB  2.15%   267KiB
 256×256×128 IsotropicDiffusivity [GPU, Float64]                          10    179ms  0.20%  17.9ms   11.6MiB  9.59%  1.16MiB
 256×256×128 SmagorinskyLilly [CPU, Float64]                              10    27.4s  31.1%   2.74s   2.73MiB  2.26%   280KiB
 256×256×128 SmagorinskyLilly [GPU, Float64]                              10    268ms  0.30%  26.8ms   12.5MiB  10.3%  1.25MiB
 256×256×128 VerstappenAnisotropicMinimumDissipation [CPU, Float64]       10    26.0s  29.5%   2.60s   2.93MiB  2.42%   300KiB
 256×256×128 VerstappenAnisotropicMinimumDissipation [GPU, Float64]       10    289ms  0.33%  28.9ms   14.0MiB  11.6%  1.40MiB
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
## Shallow Water Model

This benchmark tests the performance of the shallow water model run in a doubly periodic domain (`topology = (Periodic, Periodic, Flat)`)
on CPUs versus GPUs.  We find that we get a speed up of over 100 with `1024^2` and this almost doubles by `4096^2`.
```
Oceananigans v0.53.2
Julia Version 1.5.2
Commit 539f3ce943* (2020-09-23 23:17 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-9.0.1 (ORCJIT, cascadelake)
Environment:
  EBVERSIONJULIA = 1.5.2
  GPU: Tesla V100-SXM2-32GB

                                              Shallow water model benchmarks
┌───────────────┬─────────────┬───────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────┬─────────┐
│ Architectures │ Float_types │    Ns │        min │     median │       mean │        max │     memory │ allocs │ samples │
├───────────────┼─────────────┼───────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────┼─────────┤
│           CPU │     Float64 │    32 │   1.968 ms │   2.201 ms │   2.180 ms │   2.418 ms │ 304.19 KiB │   1879 │      10 │
│           CPU │     Float64 │    64 │   3.323 ms │   3.403 ms │   3.438 ms │   3.662 ms │ 304.28 KiB │   1885 │      10 │
│           CPU │     Float64 │   128 │   8.024 ms │   8.106 ms │   8.128 ms │   8.443 ms │ 304.28 KiB │   1885 │      10 │
│           CPU │     Float64 │   256 │  24.693 ms │  24.877 ms │  24.975 ms │  25.452 ms │ 304.28 KiB │   1885 │      10 │
│           CPU │     Float64 │   512 │  96.451 ms │  97.999 ms │  97.824 ms │  99.195 ms │ 305.09 KiB │   1937 │      10 │
│           CPU │     Float64 │  1024 │ 402.754 ms │ 404.855 ms │ 404.871 ms │ 407.238 ms │ 305.09 KiB │   1937 │      10 │
│           CPU │     Float64 │  2048 │    1.621 s │    1.625 s │    1.624 s │    1.626 s │ 305.09 KiB │   1937 │       4 │
│           CPU │     Float64 │  4096 │    6.519 s │    6.519 s │    6.519 s │    6.519 s │ 305.09 KiB │   1937 │       1 │
│           CPU │     Float64 │  8192 │   26.404 s │   26.404 s │   26.404 s │   26.404 s │ 305.09 KiB │   1937 │       1 │
│           CPU │     Float64 │ 16384 │  105.359 s │  105.359 s │  105.359 s │  105.359 s │ 305.09 KiB │   1937 │       1 │
│           GPU │     Float64 │    32 │   2.937 ms │   3.157 ms │   3.186 ms │   3.556 ms │ 637.95 KiB │   5599 │      10 │
│           GPU │     Float64 │    64 │   3.083 ms │   3.306 ms │   3.298 ms │   3.648 ms │ 657.08 KiB │   5599 │      10 │
│           GPU │     Float64 │   128 │   3.100 ms │   3.340 ms │   3.319 ms │   3.591 ms │ 691.95 KiB │   5599 │      10 │
│           GPU │     Float64 │   256 │   3.170 ms │   3.398 ms │   3.463 ms │   4.453 ms │ 767.33 KiB │   5599 │      10 │
│           GPU │     Float64 │   512 │   3.244 ms │   3.476 ms │   3.523 ms │   4.268 ms │ 912.14 KiB │   5651 │      10 │
│           GPU │     Float64 │  1024 │   3.417 ms │   3.604 ms │   3.666 ms │   4.532 ms │   1.17 MiB │   5685 │      10 │
│           GPU │     Float64 │  2048 │   9.201 ms │   9.246 ms │   9.243 ms │   9.317 ms │   1.73 MiB │   5775 │      10 │
│           GPU │     Float64 │  4096 │  35.143 ms │  35.438 ms │  35.395 ms │  35.573 ms │   2.86 MiB │   5739 │      10 │
│           GPU │     Float64 │  8192 │ 139.050 ms │ 139.118 ms │ 139.136 ms │ 139.285 ms │   5.11 MiB │   5739 │      10 │
│           GPU │     Float64 │ 16384 │ 563.804 ms │ 563.922 ms │ 563.931 ms │ 564.081 ms │   9.61 MiB │   5761 │       9 │
└───────────────┴─────────────┴───────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────┴─────────┘

        Shallow water model CPU -> GPU speedup
┌─────────────┬───────┬──────────┬─────────┬─────────┐
│ Float_types │    Ns │  speedup │  memory │  allocs │
├─────────────┼───────┼──────────┼─────────┼─────────┤
│     Float64 │    32 │ 0.697116 │ 2.09724 │ 2.97978 │
│     Float64 │    64 │  1.02927 │ 2.15944 │ 2.97029 │
│     Float64 │   128 │  2.42711 │ 2.27406 │ 2.97029 │
│     Float64 │   256 │  7.32025 │ 2.52177 │ 2.97029 │
│     Float64 │   512 │  28.1914 │ 2.98971 │  2.9174 │
│     Float64 │  1024 │  112.326 │ 3.93542 │ 2.93495 │
│     Float64 │  2048 │  175.735 │ 5.81322 │ 2.98141 │
│     Float64 │  4096 │  183.947 │ 9.58727 │ 2.96283 │
│     Float64 │  8192 │  189.795 │  17.139 │ 2.96283 │
│     Float64 │ 16384 │  186.833 │ 32.2437 │ 2.97419 │
└─────────────┴───────┴──────────┴─────────┴─────────┘

```
