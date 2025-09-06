using Test
using Random
using LinearAlgebra
using TensorKit
using GaussianfPEPS
using GaussianfPEPS: generate_cormat, cormat_blocks
using GaussianfPEPS: cormat_virtual
using SparseArrays: sparse, blockdiag

function get_J(dup::Int)
    return blockdiag((sparse([0 1; -1 0]) for _ in 1:dup)...)
end

for n in 2:6
    A = rand(2n, 2n)
    G1 = fiducial_cormat(A)
    G2 = transpose(A) * get_J(n) * A
    @test G1 ≈ G2
end

Random.seed!(0)

Np, χ = 2, 2
X, G, H, E, W = generate_cormat(Np, χ)
A, B = bogoliubov_blocks(H)
Emin = (-sum(E) + tr(A)) / 2
Emax = (sum(E) + tr(A)) / 2
@info "Lowest energy" Emin
@info "Highest energy" Emax

# blocks of Bogoliubov transformation
U, V = bogoliubov_blocks(W)
@info "Determinant of the Bogoliubov blocks U, V" det(U) det(V)
# canonical constraint
@test W * W' ≈ I
@test U * transpose(V) ≈ -V * transpose(U)
@test U * U' + V * V' ≈ I

# compare n_{ij} = ⟨a†_i a_j⟩ and x_{ij} = ⟨a_i a_j⟩
n1, x1 = cormat_to_nx(G)
n2, x2 = state_to_nx(-inv(U) * V)

@test n1 ≈ n2
@test x1 ≈ x2

# Fourier components of real correlation matrix
# G(-k)ᵀ ≈ -G(k)

A, B, D = cormat_blocks(G, Np)
for _ in 1:10
    k = rand(2) .- 0.5
    Gω1, Gω2 = cormat_virtual(-k, χ), cormat_virtual(k, χ)
    GF1 = A + B * inv(D + Gω1) * transpose(B)
    GF2 = A + B * inv(D + Gω2) * transpose(B)
    @test transpose(GF1) ≈ -GF2
end
