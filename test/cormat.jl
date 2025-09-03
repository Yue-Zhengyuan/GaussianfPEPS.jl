using Test
using Random
using LinearAlgebra
using TensorKit
using GaussianfPEPS

Random.seed!(0)

function rand_orth(n::Int; special::Bool = false)
    M = randn(Float64, (n, n))
    F = qr(M)
    Q = Matrix(F.Q)
    R = F.R
    # absorb signs of diag(R) into Q
    λ = diag(R) ./ abs.(diag(R))
    Q .= Q .* λ'
    if special
        # ensure det(Q)=+1
        if det(Q) < 0
            Q[:, 1] .*= -1
        end
    end
    return Q
end

function generate_cormat(Np::Int, χ::Int)
    N = Np + 4χ
    while true
        X = rand_orth(2N)
        G = fiducial_cormat(X)
        H = parent_Hamiltonian_BdG(G)
        E, W = bogoliubov(H)
        if det(W) ≈ 1
            return X, G, H, E, W
        end
    end
    return
end

Np, χ = 2, 2
X, G, H, E, W = generate_cormat(Np, χ)
A, B = bogoliubov_blocks(H)
Emin = (-sum(E) + tr(A)) / 2
Emax = (sum(E) + tr(A)) / 2
@info "Lowest energy" Emin
@info "Highest energy" Emax

# blocks of Bogoliubov transformation
U, V = bogoliubov_blocks(W)
@show (det(U), det(V))
# canonical constraint
@assert W * W' ≈ I
@assert U * transpose(V) ≈ -V * transpose(U)
@assert U * U' + V * V' ≈ I

# compare n_{ij} = ⟨a†_i a_j⟩ and x_{ij} = ⟨a_i a_j⟩
n1, x1 = cormat_to_nx(G)
n2, x2 = state_to_nx(-inv(U) * V)

@test n1 ≈ n2
@test x1 ≈ x2
