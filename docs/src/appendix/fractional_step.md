# Fractional step method

In some models (e.g., `IncompressibleModel` or `HydrostaticFreeSurfaceModel`) solving the momentum 
coupled with the continuity equation can be cumbersome so instead we employ a fractional step 
method. To approximate the solution of the coupled system we first solve an approximation to 
the discretized momentum equation for an intermediate velocity field ``\boldsymbol{v}^\star`` 
without worrying about satisfying the incompressibility constraint. We then project ``\boldsymbol{v}^\star`` 
onto the space of divergence-free velocity fields to obtain a value for ``\boldsymbol{v}^{n+1}`` 
that satisfies continuity.

For example, for the `IncompressibleModel`, if we ignore the background velocity fields and the
surface waves, we thus discretize the momentum equation as
```math
  \frac{\boldsymbol{v}^\star - \boldsymbol{v}^n}{\Delta t}
    = - \left[ \boldsymbol{v} \boldsymbol{\cdot} \boldsymbol{\nabla} \boldsymbol{v} \right]^{n+\frac{1}{2}}
      - \boldsymbol{f} \times \boldsymbol{v}^{n+\frac{1}{2}}
      + \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \nu \boldsymbol{\nabla} \boldsymbol{v}^{n+\frac{1}{2}} \right )
      + \boldsymbol{F}_{\boldsymbol{v}}^{n+\frac{1}{2}} \, ,
```
where the superscript ``n + \frac{1}{2}`` indicates that these terms are evaluated at time step 
``n + \frac{1}{2}``, which we compute explicitly (see [Time Stepping section](../numerical_implementation/time_stepping)).

The projection is then performed
```math
   \boldsymbol{v}^{n+1} = \boldsymbol{v}^\star - \Delta t \, \boldsymbol{\nabla} p^{n+1} \, ,
```
to obtain a divergence-free velocity field ``\boldsymbol{v}^{n+1}``. Here the projection is performed by solving an elliptic
problem for the pressure ``p^{n+1}`` with the boundary condition
```math
  \boldsymbol{\hat{n}} \boldsymbol{\cdot} \boldsymbol{\nabla} p^{n+1} |_{\partial\Omega} = 0 \, .
```

[Orszag86](@cite) and [Brown01](@cite) raise an important issue regarding these fractional step 
methods, which is that "*while the velocity can be reliably computed to second-order accuracy 
in time and space, the pressure is typically only first-order accurate in the ``L_\infty``-norm.*" 
The numerical boundary conditions must be carefully accounted for to ensure the second-order 
accuracy promised by the fractional step methods.
