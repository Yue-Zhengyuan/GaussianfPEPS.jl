"""
Direct sum of [0 1; -1 0] for `dup` times.
"""
function get_J(dup::Int)
    return blockdiag(ntuple(_ -> iσy, dup)...)
end

"""
Direct sum of `[1 im] / 2` for `dup` times
"""
function get_W(dup::Int)
    return blockdiag(ntuple(_ -> w, dup)...)
end

"""
Construct the fiducial state local correlation matrix
`Γ = Xᵀ J X`, where `X` is a real orthogonal matrix, 
and `J` is the direct sum of `[0 1; -1 0]` blocks.
"""
function fiducial_cormat(X::AbstractMatrix)
    @assert eltype(X) <: Real && X' * X ≈ I
    J = get_J(div(size(X, 1), 2))
    return transpose(X) * J * X
end

"""
Get the blocks of (Np + 4*χ)-dimensional real correlation matrix 
```
    G = [A B; -Bᵀ D]
```
"""
function cormat_blocks(G::AbstractMatrix, Np::Int = 2)
    @assert eltype(G) <: Real && G ≈ -transpose(G)
    A = G[1:(2 * Np), 1:(2 * Np)]
    B = G[1:(2 * Np), (2 * Np + 1):end]
    D = G[(2 * Np + 1):end, (2 * Np + 1):end]
    return A, B, D
end

"""
Calculate the 2-body expectation value matrices
`n_{ij} = ⟨a†_i a_j⟩` and `x_{ij} = ⟨a_i a_j⟩`
using the real correlation matrix `G`
"""
function cormat_to_nx(G::AbstractMatrix)
    @assert eltype(G) <: Real && G ≈ -transpose(G)
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
