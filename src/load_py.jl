"""
Load the orthogonal matrix U produced by 
https://github.com/TensorBFS/Gaussian-fPEPS
for fermion bilinear (BCS) Hamiltonian.

The input order of physical/virtual complex fermions is
(p_1, ..., p_{Np}, r_1, l_1, ..., r_χ, l_χ, d_1, u_1, ..., d_χ, u_χ)
(as given in Physical Review B 107, 125128 (2023)).

Note that the x/y-direction points rightwards/downwards.

The dimension of U is 2 * (Np + 4χ) (on square lattice).

In terms of Majorana fermions, each complex fermion is replaced by 
(c1, c2).
"""
function load_orth(file::AbstractString)
    data = h5open(file, "cw")
    U = data["transformer/T"][]
    return U
end
