using Test
using Random
using LinearAlgebra
using TensorKit
using TensorKitTensors.FermionOperators
using GaussianfPEPS

Random.seed!(0)

# create Hamiltonian
N = 6
A = randn(ComplexF64, (N, N))
A = A + A'
B = randn(ComplexF64, (N, N))
B = B - transpose(B)
v = randn(reduce(⊗, fill(fermion_space(), N)))

ham = parent_Hamiltonian(A, B)
v1 = ham * v
v2 = parent_Hamiltonian_map(A, B, v)
@test v1 ≈ v2
