"""
Create the vacuum state for `n` spinless fermions
"""
function vacuum_state(T::Type{<:Number}, n::Int)
    vac = zeros(T, V)
    vac.data[1] = 1.0
    return (n > 1) ? reduce(⊗, fill(vac, n)) : vac
end
vacuum_state(n::Int) = vacuum_state(ComplexF64, n)

"""
Construct the maximally entangled state (MES) on virtual bonds
for χ pairs of virtual fermions `(a1_i, a2_i)` (i = 1, ..., χ)
```
    |ω⟩ = ∏_{i=1}^χ 2⁻½ (1 + a1†_i a2†_i) |0⟩
```
"""
function virtual_state(T::Type{<:Number}, χ::Int)
    ff = FO.f_plus_f_plus(T)
    vac = vacuum_state(T, 2)
    # MES for one pair of (a1_i, a2_i) on the bond
    # the resulting fermion order is (a1_1, a2_1, ..., a1_χ, a2_χ)
    ω = (1 / sqrt(2)) * (unit ⊗ unit + ff) * vac
    if χ > 1
        # reorder fermions to (a1_1, ..., a1_χ, a2_1, ..., a2_χ)
        ω = reduce(⊗, fill(ω, χ))
        perm = Tuple(vcat(1:2:(2χ), 2:2:(2χ)))
        ω = permute(ω, (perm, ()))
    end
    return ω
end
virtual_state(χ::Int) = virtual_state(ComplexF64, χ)

"""
Construct the fully paired state `exp(a† A a† / 2)`, 
where A is an anti-symmetric matrix.
"""
function paired_state(T::Type{<:Number}, A::AbstractMatrix)
    N = size(A, 1)
    @assert A ≈ -transpose(A)
    ff = FO.f_plus_f_plus(T)
    ψ = vacuum_state(T, N)
    # apply exp(A_{ij} a†_i a†_j) (i < j)
    for i in 1:(N - 1)
        for j in (i + 1):N
            op = exp(A[i, j] * ff)
            idx_op = [-i, -j, i, j]
            idx_ψ = collect(-1:-1:(-N))
            idx_ψ[i], idx_ψ[j] = i, j
            ψ = ncon([op, ψ], [idx_op, idx_ψ])
        end
    end
    return ψ
end
paired_state(A) = paired_state(ComplexF64, A)

"""
Construct the local tensor of the fiducial state
`exp(a† A a† / 2)`, where A is an anti-symmetric matrix.

Input complex fermion order in `a` should be
(p_1, ..., p_{Np}, l_1, r_1, ..., l_χ, r_χ, d_1, u_1, ..., d_χ, u_χ)

The output complex fermion order will be
(p_1, ..., p_{Np}, l_1, ..., l_χ, r_1, ..., r_χ, d_1, ..., d_χ, u_1, ..., u_χ)
"""
function fiducial_state(T::Type{<:Number}, Np::Int, χ::Int, A::AbstractMatrix)
    ψ = paired_state(T, A)
    # reorder virtual fermions
    perm = vcat(1:2:(2χ), 2:2:(2χ))
    perm = Tuple(vcat(1:Np, perm .+ Np, perm .+ (Np + 2χ)))
    ψ = permute(ψ, (perm, ()))
    return ψ
end
function fiducial_state(Np::Int, χ::Int, A::AbstractMatrix)
    return fiducial_state(ComplexF64, Np, χ, A)
end

"""
Get PEPS tensor by contracting virtual axes of ⟨ω|F⟩,
where |ω⟩, |F⟩ are the virtual and the fiducial states.
```
            -2
            ↓
            ω
            ↑
            1  -1
            ↑ ↗
    -5  --←-F-→- 2 -→-ω-←- -3
            ↓
            -4
```
Input axis order
```
        5  1                2
        ↑ ↗                 ↑
    2-←-F-→-3   1-←-ω-→-2   ω
        ↓                   ↓
        4                   1
```
"""
function get_peps(ω::AbstractTensor{T, S, N1}, F::AbstractTensor{T, S, N2}) where {T, S, N1, N2}
    χ = div(N1, 2)
    Np = N2 - 4χ
    # merge physical and virtual axes
    fuser_p = isomorphism(fuse(fill(V, Np)...), reduce(⊗, fill(V, Np)))
    fuser_v = isomorphism(fuse(fill(V, χ)...), reduce(⊗, fill(V, χ)))
    ω = (fuser_v ⊗ fuser_v) * ω
    F = (fuser_p ⊗ reduce(⊗, fill(fuser_v, 4))) * F
    @tensor A[-1; -2 -3 -4 -5] := conj(ω[1 -2]) * conj(ω[2 -3]) * F[-1 -5 2 -4 1]
    return InfinitePEPS(A; unitcell = (1, 1))
end

"""
Translate the orthogonal matrix `X` produced by Gaussian-fPEPS
to Gaussian fPEPS in PEPSKit format.
"""
function translate(X::AbstractMatrix, Np::Int, χ::Int)
    N = Np + 4χ
    @assert size(X, 1) == 2N && X' * X ≈ I
    G = fiducial_cormat(X)
    H = parent_Hamiltonian_BdG(G)
    E, W = bogoliubov(H)
    detW = det(W)
    if !(detW ≈ 1.0)
        @assert detW ≈ -1
        error("det(W) = -1; fiducial state has odd parity.")
    end
    A, B = bogoliubov_blocks(W)
    ω = virtual_state(χ)
    F = fiducial_state(Np, χ, -inv(A) * B)
    peps = get_peps(ω, F)
    return peps
end
