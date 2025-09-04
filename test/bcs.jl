using Test
using Random
using LinearAlgebra
using GaussianfPEPS

# doping - mu relation
Lx, Ly = 128, 128
t, mu, Δx, Δy = 1.0, -0.6, 0.5, -0.5
for pbcs in Iterators.product((true, false), (true, false))
    bz = BrillouinZone((Lx, Ly), pbcs)
    δ = BCS.doping_exact(bz; t, mu, Δx, Δy)
    mu′ = BCS.solve_mu(bz, δ; t, Δx, Δy)
    @test mu ≈ mu′
end
