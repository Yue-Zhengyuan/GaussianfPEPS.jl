"""
Check if a 2-site bond is a nearest neighbor x-bond
"""
function _is_xbond(bond)
    return bond[2] - bond[1] == CartesianIndex(0, 1)
end

"""
Generate a random real orthogonal matrix
"""
function rand_orth(n::Int; special::Bool = false)
    M = randn(Float64, (n, n))
    F = qr(M)
    Q = Matrix(F.Q)
    R = F.R
    # absorb signs of diag(R) into Q
    λ = diag(R) ./ abs.(diag(R))
    Q .= Q .* λ'
    if special
        # ensure det(Q)=+1
        if det(Q) < 0
            Q[:, 1] .*= -1
        end
    end
    return Q
end

"""
Generate a random correlation matrix `G` of 
a pure Gaussian state with even parity
"""
function generate_cormat(Np::Int, χ::Int)
    N = Np + 4χ
    while true
        X = rand_orth(2N)
        G = fiducial_cormat(X)
        H = parent_Hamiltonian_BdG(G)
        E, W = bogoliubov(H)
        if det(W) ≈ 1
            return X, G, H, E, W
        end
    end
    return
end
