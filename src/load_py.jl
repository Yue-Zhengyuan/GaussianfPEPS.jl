"""
Load the orthogonal matrix U produced by 
https://github.com/TensorBFS/Gaussian-fPEPS
for spin-1/2 fermion BCS Hamiltonian.

The dimension of U is 2 * (Np + 4χ) (on square lattice).

The input order of physical/virtual complex fermions is (Np = 2)
(p_1, ..., p_{Np}, r_1, l_1, ..., r_χ, l_χ, u_1, d_1, ..., u_χ, d_χ).
In terms of Majorana fermions, each complex fermion is replaced by 
(c1, c2).

We need to transform the order to
(p_1, ..., p_{Np}, l_1, r_1, ..., l_χ, r_χ, d_1, u_1, ..., d_χ, u_χ).
"""
function load_orth(file::AbstractString)
    data = h5open(file, "cw")
    U = data["transformer/T"][]
    # reorder virtual Majorana fermions
    Np = 2
    χ = div(size(U, 1) - 2 * Np, 4)
    phyperm = collect(1:(2 * Np))
    virperm = [x + 2 * Np for n in 1:χ for x in (4n - 1, 4n, 4n - 3, 4n - 2)]
    perm = vcat(phyperm, virperm)
    return U[perm, perm]
end
