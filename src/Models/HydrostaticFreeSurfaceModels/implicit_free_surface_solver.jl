using Oceananigans.Solvers
using Oceananigans.Operators

"""
    implicit_free_surface_linear_operation!(result, x, arch, grid, bcs; args...)

Returns `L(ηⁿ)`, where `ηⁿ` is the free surface displacement at time step `n`
and `L` is the linear operator that arises
in an implicit time step for the free surface displacement `η`.

(See the docs section on implicit time stepping.)
"""
function implicit_free_surface_linear_operation!(L_ηⁿ⁺¹, ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)
    grid = L_ηⁿ⁺¹.grid
    arch = architecture(L_ηⁿ⁺¹)

    fill_halo_regions!(ηⁿ⁺¹)

    event = launch!(arch, grid, :xy, _implicit_free_surface_linear_operation!,
                    L_ηⁿ⁺¹, grid,  ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt,
                    dependencies=Event(device(arch)))

    wait(device(arch), event)

    fill_halo_regions!(L_ηⁿ⁺¹)

    return nothing
end

# Kernels that act on vertically integrated / surface quantities
@inline ∫ᶻ_Ax_∂x_ηᶠᶜᶜ(i, j, k, grid, ∫ᶻ_Axᶠᶜᶜ, η) = @inbounds ∫ᶻ_Axᶠᶜᶜ[i, j, k] * ∂xᶠᶜᵃ(i, j, k, grid, η)
@inline ∫ᶻ_Ay_∂y_ηᶜᶠᶜ(i, j, k, grid, ∫ᶻ_Ayᶜᶠᶜ, η) = @inbounds ∫ᶻ_Ayᶜᶠᶜ[i, j, k] * ∂yᶜᶠᵃ(i, j, k, grid, η)

"""
Compute the horizontal divergence of vertically-uniform quantity using
vertically-integrated face areas `∫ᶻ_Axᶠᶜᶜ` and `∫ᶻ_Ayᶜᶠᶜ`.
"""
@inline ∇h²ᶜᶜᵃ(i, j, k, grid, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, η::ReducedField{X, Y, Nothing}) where {X, Y} =
    1 / Azᶜᶜᵃ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, ∫ᶻ_Ax_∂x_ηᶠᶜᶜ, ∫ᶻ_Axᶠᶜᶜ, η) +
                                δyᵃᶜᵃ(i, j, k, grid, ∫ᶻ_Ay_∂y_ηᶜᶠᶜ, ∫ᶻ_Ayᶜᶠᶜ, η))

"""
    _implicit_free_surface_linear_operation!(L_ηⁿ⁺¹, grid, ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)

Returns the left hand side of the "implicit η equation"

```math
(∇ʰ⋅H∇ʰ - 1/gΔt²) ηⁿ⁺¹ = 1 / (gΔt) ∇ʰH U̅ˢᵗᵃʳ - 1 / (gΔt²) ηⁿ
----------------------
        ≡ L_ηⁿ⁺¹
```

which is written in a discrete finite volume form in which the equation
is arranged to ensure a symmtric form by multiplying by horizontal areas Az:

```
δⁱÂʷ∂ˣηⁿ⁺¹ + δʲÂˢ∂ʸηⁿ⁺¹ - 1/gΔt² Az ηⁿ⁺¹ = 1 / (gΔt) (δⁱÂʷu̅ˢᵗᵃʳ + δʲÂˢv̅ˢᵗᵃʳ) - 1 / gΔt² Az ηⁿ
```

where  ̂ indicates a vertical integral, and                   
       ̅ indicates a vertical average                         
"""
@kernel function _implicit_free_surface_linear_operation!(L_ηⁿ⁺¹, grid, ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)
    i, j = @index(Global, NTuple)

    @inbounds L_ηⁿ⁺¹[i, j, 1] = (  Azᶜᶜᵃ(i, j, 1, grid) * ∇h²ᶜᶜᵃ(i, j, 1, grid, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, ηⁿ⁺¹)
                                 - Azᶜᶜᵃ(i, j, 1, grid) * ηⁿ⁺¹[i, j, 1] / (g * Δt^2))
end
