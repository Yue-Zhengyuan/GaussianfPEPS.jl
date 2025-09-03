"""
Construct the direct sum of [1 1; im -im] for `dup` times.
"""
function get_W2(dup::Int)
    return blockdiag((w2 for _ in 1:dup)...)
end

"""
Given a real correlation matrix G of Majorana fermions, construct 
the BdG matrix of the parent Hamiltonian in terms of complex fermions.
"""
function parent_Hamiltonian_BdG(G::AbstractMatrix)
    @assert eltype(G) <: Real && G ≈ -transpose(G)
    N = div(size(G, 1), 2)
    # change to complex fermion basis
    # c_{2j-1} = f_j + f†_j, c_{2i} = i(f_j - f†_j)
    # resulting fermion order is (f_1, f†_1, ..., f_N, f†_N)
    W = get_W2(N)
    H = W' * (-0.5im * G) * W
    # put annihilation in front of creation operators
    # (f_1, ..., f_N, f†_1, ..., f†_N)
    perm = vcat(1:2:(2N), 2:2:(2N))
    return Hermitian(H[perm, perm])
end

"""
Extract the blocks A, B of bilinear fermion Hamiltonian H
`H = [A B; -B̄ -Ā]`, where `A = A'` and `Bᵀ = -B`; 
or U, V of the Bogoliubov transformation `W = [U V; V̄ Ū]`
"""
function bogoliubov_blocks(H::AbstractMatrix)
    N = div(size(H, 1), 2)
    return H[1:N, 1:N], H[1:N, (N + 1):end]
end

"""
Get the length-N permutation list that moves the 1st, 2nd
elements to the i-th and j-th position (i ≠ j)
"""
function _get_perm(N::Int, i::Int, j::Int)
    @assert 1 <= i <= N && 1 <= j <= N && i != j
    p = zeros(Int, N)
    p[1], p[2] = i, j
    rem = filter(k -> k != i && k != j, 1:N)
    @inbounds for k in 3:N
        p[k] = rem[k - 2]
    end
    return invperm(Tuple(p))
end

"""
Given the blocks A, B in the BdG matrix `H = [A B; -B̄ -Ā]` of a 
fermion bilinear Hamiltonian, construct the Hamiltonian operator 
```
    H = ∑_{i,j} (A_{ij} a†_i a_j + (1/2) B_{ij} a†_i a†_j - (1/2) B̄_{ij} a_i a_j)
    = ∑_i A_{ii} a†_i a_j
        + ∑_{i < j} (A_{ij} a†_i a_j - A_{ji} a_i a†_j)
        + ∑_{i < j} (B_{ij} a†_i a†_j - B̄_{ij} a_i a_j)
```
where A is Hermitian, and B is anti-symmetric.
"""
function bilinear_Hamiltonian(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: Number}
    @assert A ≈ A' && B ≈ -transpose(B)
    @assert size(A) == size(B)
    N = size(A, 1)
    pspace = reduce(⊗, fill(V, N))
    H = zeros(T, pspace → pspace)
    num = FO.f_num(T)
    pm = FO.f_plus_f_min(T)
    mp = FO.f_min_f_plus(T)
    pp = FO.f_plus_f_plus(T)
    mm = FO.f_min_f_min(T)
    # i = j terms
    @inbounds for i in 1:N
        op = A[i, i] * reduce(⊗, (s == i ? num : unit) for s in 1:N)
        H += op
    end
    # i ≠ j terms
    @inbounds for i in 1:(N - 1)
        @inbounds for j in (i + 1):N
            op = A[i, j] * pm - A[j, i] * mp + B[i, j] * pp - conj(B[i, j]) * mm
            if N > 2
                # permute axes (1, 2) to (i, j)
                op = op ⊗ reduce(⊗, fill(unit, N - 2))
                perm1 = _get_perm(N, i, j)
                perm2 = perm1 .+ N
                H += permute(op, (perm1, perm2))
            else
                H += op
            end
        end
    end
    @assert H ≈ H'
    return H
end

"""
Construct the fermion bilinear Hamiltonian with BdG matrix 
`H = [A B; -B̄ -Ā]` as a linear map on input state `v`, 
which may have an additional auxiliary leg.
"""
function bilinear_Hamiltonian_map(
        A::AbstractMatrix{T}, B::AbstractMatrix{T}, v::AbstractTensor
    ) where {T <: Number}
    @assert A ≈ A' && B ≈ -transpose(B)
    @assert size(A) == size(B)
    N = size(A, 1)
    # v may carry additional auxiliary 1-dimensional legs to
    # allow nonzero charges or odd fermion parity
    Nv = numout(v)
    @assert Nv >= N
    num = FO.f_num(T)
    pm = FO.f_plus_f_min(T)
    mp = FO.f_min_f_plus(T)
    pp = FO.f_plus_f_plus(T)
    mm = FO.f_min_f_min(T)
    v2 = similar(v)
    v2.data .= 0.0
    # i = j terms
    @inbounds for i in 1:N
        op = A[i, i] * num
        idx_v = collect(-1:-1:(-Nv))
        idx_v[i] = i
        v2 += ncon([op, v], [[-i, i], idx_v])
    end
    # i ≠ j terms
    @inbounds for i in 1:(N - 1)
        @inbounds for j in (i + 1):N
            op = A[i, j] * pm - A[j, i] * mp + B[i, j] * pp - conj(B[i, j]) * mm
            idx_v = collect(-1:-1:(-Nv))
            idx_v[i], idx_v[j] = i, j
            v2 += ncon([op, v], [[-i, -j, i, j], idx_v])
        end
    end
    return v2
end

"""
Bogoliubov transformation of a fermionic bilinear Hamiltonian `H`. Returns 
- The (positive) energy spectrum `E`, in descending order;
- The transformation `W = [U V; V̄ Ū]` (such that `W * H * W' = diagm(vcat(E, -E))`);
"""
function bogoliubov(H::Hermitian)
    N = size(H, 1)
    E, W0 = eigen(H; sortby = (x -> -real(x)))
    n = div(N, 2)
    # construct the transformation W
    Wpos = W0[:, 1:n]
    U = Wpos[1:n, :]
    V = conj(Wpos[(n + 1):end, :])
    W = similar(W0)
    W[1:n, 1:n] = U
    W[1:n, (n + 1):(2n)] = V
    W[(n + 1):(2n), 1:n] = conj.(V)
    W[(n + 1):(2n), (n + 1):(2n)] = conj.(U)
    # check canonical constraint
    @assert W' * W ≈ I
    # check positiveness of energy
    @assert all(E[1:n] .> 0)
    return E[1:n], Matrix(W')
end
