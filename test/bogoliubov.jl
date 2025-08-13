using Test
using Random
using LinearAlgebra
using GaussianfPEPS

Random.seed!(0)

# create Hamiltonian
N = 6
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
@show det(W)
@show E
@test W * H * W' ≈ diagm(vcat(E, -E))
Emin = (-sum(E) + tr(A)) / 2
@show Emin

# Hamiltonian operator
ham = bilinear_Hamiltonian(A, B)
D, U = eigen(ham)
@test minimum(real(D.data)) ≈ Emin
