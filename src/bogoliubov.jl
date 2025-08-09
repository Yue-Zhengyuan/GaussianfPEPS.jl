"""
Construct the direct sum of [1 1; im -im] for `dup` times.
"""
function get_W2(dup::Int)
    return blockdiag(ntuple(_ -> w2, dup)...)
end

"""
Given a real correlation matrix G of Majorana fermions, construct 
the parent Hamiltonian in terms of complex fermions.
"""
function parent_Hamiltonian(G::AbstractMatrix)
    @assert eltype(G) <: Real && G ≈ -transpose(G)
    N = div(size(G, 1), 2)
    # change to complex fermion basis
    # c_{2j-1} = f_j + f†_j, c_{2i} = i(f_j - f†_j)
    # resulting fermion order is (f_1, f†_1, ..., f_N, f†_N)
    W = get_W2(N)
    H = W' * (-0.5im * G) * W
    # put annhilation in front of creation operators
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
Bogoliubov transformation of a fermionic bilinear Hamiltonian `H`. Returns 
- The (positive) energy spectrum `E`, in descending order;
- The transformation `W = [U V; V̄ Ū]` (such that `W' * H * W = diagm(vcat(E, -E))`);
- The ground state matrix `A = -inv(U) * V`. 
"""
function bogoliubov(H::Hermitian)
    N = size(H, 1)
    E, W0 = eigen(H; sortby=(x -> -real(x)))
    n = div(N, 2)
    # construct the transformation W
    Wpos = W0[:, 1:n]
    U = Wpos[1:n, :];
    V = conj(Wpos[(n+1):end, :]);
    W = similar(W0)
    W[1:n, 1:n] = U
    W[1:n, n+1:2n] = V
    W[n+1:2n, 1:n] = conj.(V)
    W[n+1:2n, n+1:2n] = conj.(U)
    # the ground state is exp(a† A a† / 2), where A = -inv(U) * V
    A = -inv(U) * V
    return E[1:n], W, A
end
