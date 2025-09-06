"""
Direct sum of [0 1; -1 0] for `dup` times.
"""
function get_J(dup::Int)
    return blockdiag((iσy for _ in 1:dup)...)
end

"""
Direct sum of `[1 im] / 2` for `dup` times
"""
function get_W(dup::Int)
    return blockdiag((w for _ in 1:dup)...)
end

"""
Fourier components of the virtual state correlation matrix,
with χ species of complex virtual fermions along each direction.

The virtual Majorana fermions are ordered as
(l_1, r_1, ..., l_χ, r_χ, d_1, u_1, ..., d_χ, u_χ)
"""
function cormat_virtual(k::Vector{Float64}, χ::Int)
    expx, expy = cispi(2 * k[1]), cispi(2 * k[2])
    xmat = sparse(1:1:4, 4:-1:1, [-1 / expx, -1 / expx, expx, expx], 4, 4)
    ymat = sparse(1:1:4, 4:-1:1, [-1 / expy, -1 / expy, expy, expy], 4, 4)
    return blockdiag(((n <= χ ? xmat : ymat) for n in 1:2χ)...)
end

"""
Construct the fiducial state local correlation matrix
`Γ = Xᵀ J X`, where `X` is a real orthogonal matrix, 
and `J` is the direct sum of `[0 1; -1 0]` blocks.

It is user's responsibility to ensure the orthogonality of input `X`.
"""
function fiducial_cormat(X::Matrix{Float64})
    J = get_J(div(size(X, 1), 2))
    return transpose(X) * J * X
end

"""
Get the blocks of (Np + 4*χ)-dimensional real correlation matrix 
```
    G = [A B; -Bᵀ D]
```
"""
function cormat_blocks(G::Matrix{Float64}, Np::Int = 2)
    @assert G ≈ -transpose(G)
    return G[1:(2 * Np), 1:(2 * Np)],
        G[1:(2 * Np), (2 * Np + 1):end],
        G[(2 * Np + 1):end, (2 * Np + 1):end]
end

"""
Calculate the 2-body expectation value matrices
`n_{ij} = ⟨a†_i a_j⟩` and `x_{ij} = ⟨a_i a_j⟩`
using the real correlation matrix `G`
"""
function cormat_to_nx(G::Matrix{Float64})
    @assert G ≈ -transpose(G)
    dup = div(size(G, 1), 2)
    W = get_W(dup)
    n = I(dup) / 2 - 1.0im * W * G * adjoint(W)
    x = -1.0im * conj(W) * G * adjoint(W)
    return n, x
end

"""
Calculate the 2-body expectation value matrices
`n_{ij} = ⟨a†_i a_j⟩` and `x_{ij} = ⟨a_i a_j⟩`
in the Gaussian state specified by `exp(a† A a† / 2)`,
where A is a complex antisymmetric matrix.
"""
function state_to_nx(A::AbstractMatrix)
    @assert A ≈ -transpose(A)
    X = -I - A' * A
    n = inv(X) + I
    x = A * inv(X)
    return n, x
end
