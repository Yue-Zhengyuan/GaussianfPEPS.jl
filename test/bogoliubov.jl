using Test
using Random
using LinearAlgebra
using TensorKit
using GaussianfPEPS
using TensorKitTensors.FermionOperators: fermion_space

Random.seed!(0)

function generate_bdg(N::Int)
    # keep trying until the ground state has even fermion parity
    while true
        # create Hamiltonian
        A = randn(ComplexF64, (N, N))
        A = A + A'
        B = randn(ComplexF64, (N, N))
        B = B - transpose(B)
        H = zeros(ComplexF64, (2N, 2N))
        H[1:N, 1:N], H[1:N, (N + 1):2N] = A, B
        H[(N + 1):2N, 1:N], H[(N + 1):2N, (N + 1):2N] = -conj(B), -conj(A)
        H = Hermitian(H)
        # bogoliubov transformation
        E, W = bogoliubov(H)
        if det(W) ≈ 1.0
            return A, B, H, E, W
        end
    end
    return
end

N = 6
A, B, H, E, W = generate_bdg(N)
@test W * H * W' ≈ diagm(vcat(E, -E))
Emin = (-sum(E) + tr(A)) / 2
@info "Ground state energy" Emin

# create the (even parity) ground state
X, Y = bogoliubov_blocks(W)
gs = paired_state(-inv(X) * Y)

# Hamiltonian operator
ham = bilinear_Hamiltonian(A, B)
D, U = eigen(ham)
# the minimum energy (should be in even parity block)
@test minimum(block(D, FermionParity(0))) ≈ Emin
@test ham * gs ≈ Emin * gs

# Hamiltonian map
v = randn(reduce(⊗, fill(fermion_space(), N)))
v1 = ham * v
v2 = bilinear_Hamiltonian_map(A, B, v)
@test v1 ≈ v2
@test bilinear_Hamiltonian_map(A, B, gs) ≈ Emin * gs
