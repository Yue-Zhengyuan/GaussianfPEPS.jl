function _is_xbond(bond)
    return bond[2] - bond[1] == CartesianIndex(0, 1)
end

"""
BCS spin-1/2 Hamiltonian with singlet pairing terms on square lattice
```
    H = -t ∑_{i,v} (c†_{iα} c_{i+v,α} + h.c.) - μ ∑_i c†_{iα} c_{iα}
        + ∑_{i,v} (Δv ϵ_{αβ} c†_{iα} c†_{i+v,β} + h.c.)
```
where v sums over the basis vectors e_x, e_y. 

- s-wave state: Δx = Δy.
- d-wave state: Δx = -Δy.
"""
function bcs_spinhalf(
        T::Type{<:Number}, lattice::InfiniteSquare; t::Float64 = 1.0,
        Δx::Float64 = 0.5, Δy::Float64 = -0.5, mu::Float64 = 0.0
    )
    pspace = HO.hubbard_space(Trivial, Trivial)
    pspaces = fill(pspace, (lattice.Nrows, lattice.Ncols))
    num = HO.e_num(T, Trivial, Trivial)
    unit = TensorKit.id(T, pspace)
    hopping = (-t) * HO.e_hopping(T, Trivial, Trivial) -
        (mu / 4) * (num ⊗ unit + unit ⊗ num)
    pairing = HO.singlet_plus(T, Trivial, Trivial)
    pairing += pairing'
    return LocalOperator(
        pspaces,
        map(nearest_neighbours(lattice)) do bond
            return bond => hopping + pairing * (_is_xbond(bond) ? Δx : Δy)
        end...
    )
end
bcs_spinhalf(lattice; t, Δx, Δy, μ) = bcs_spinhalf(ComplexF64, lattice; t, Δx, Δy, μ)

"""
The exact energy of a Gaussian fPEPS (specified by the real correlation matrix 
`G` of the fiducial state, with `Np` physical fermions) on a finite lattice 
specified by the BrillouinZone `bz`.
"""
function bcs_energy(
        G::AbstractMatrix, bz::BrillouinZone, Np::Int;
        t::Float64 = 1.0, Δx::Float64 = 0.5, Δy::Float64 = -0.5, mu::Float64 = 0.0
    )
    A, B, D = cormat_blocks(G, Np)
    χ = div(size(G, 1) - 2 * Np, 8)
    energy = 0.0
    for k in bz.ks
        Gω = cormat_virtual(k, χ)
        Gf = A + B * inv(D + Gω) * transpose(B)
        ξ = -2t * (cos(2π * k[1]) + cos(2π * k[2])) - mu
        Δ = 2Δx * cos(2π * k[1]) + 2Δy * cos(2π * k[2])
        energy += ξ * (2 - Gf[1, 2] - Gf[3, 4]) / 2
        energy += (-1 / 2) * real(
            Δ * (Gf[4, 1] + Gf[3, 2] + 1.0im * (Gf[4, 2] - Gf[3, 1]))
        )
    end
    return real(energy) / prod(size(bz))
end
