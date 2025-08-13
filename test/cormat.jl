using Test
using HDF5
using Random
using LinearAlgebra
using TensorKit
using GaussianfPEPS

Np, χ = 2, 2
N = Np + 4χ
# load correlation matrix produced by Gaussian_fPEPS
data = h5open("test/input/data.h5", "r")
Q = data["transformer/T"][]
# continue to build the fiducial Hamiltonian
G = fiducial_cormat(Q)
@assert size(G, 1) == 2N
H = parent_Hamiltonian_BdG(G)

# Bogoliubov transformation
E, W = bogoliubov(H);
A, B = bogoliubov_blocks(H);
# when det W = +1, the ground state has even parity
@assert det(W) ≈ 1
Emin = (-sum(E) + tr(A)) / 2
Emax = (sum(E) + tr(A)) / 2
@show (Emin, Emax);

# blocks of Bogoliubov transformation
U, V = bogoliubov_blocks(W);
@show (det(U), det(V));
# canonical constraint
@assert W * W' ≈ I
@assert U * transpose(V) ≈ -V * transpose(U)
@assert U * U' + V * V' ≈ I

# compare n_{ij} = ⟨a†_i a_j⟩ and x_{ij} = ⟨a_i a_j⟩
n1, x1 = cormat_to_nx(G)
n2, x2 = state_to_nx(-inv(U) * V);

@test n1 ≈ n2
@test x1 ≈ x2
