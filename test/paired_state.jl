using Test
using Random
using TensorKit
using LinearAlgebra
using GaussianfPEPS
import TensorKitTensors.FermionOperators as FO

Random.seed!(0)
for N in (4, 5)
    A = randn(ComplexF64, (N, N))
    A = A - transpose(A)
    ψ = paired_state(A)
    normalize!(ψ)
    n, x = state_to_nx(A)

    # ⟨a_i a_j⟩
    x2 = similar(x)
    op = FO.f_min_f_min()
    for idx in CartesianIndices(x)
        i, j = Tuple(idx)
        if i == j
            x2[idx] = 0.0
        else
            idx_op = [-i, -j, i, j]
            idx_ψ = collect(-1:-1:(-N))
            idx_ψ[i], idx_ψ[j] = i, j
            ψ2 = ncon([op, ψ], [idx_op, idx_ψ])
            x2[idx] = (ψ' * ψ2).data[1]
        end
    end
    @test x2 ≈ x

    # ⟨a†_i a_j⟩
    n2 = similar(n)
    op = FO.f_plus_f_min()
    op1 = FO.f_num()
    for idx in CartesianIndices(x)
        i, j = Tuple(idx)
        if i == j
            idx_op = [-i, i]
            idx_ψ = collect(-1:-1:(-N))
            idx_ψ[i] = i
            ψ2 = ncon([op1, ψ], [idx_op, idx_ψ])
            n2[idx] = (ψ' * ψ2).data[1]
            continue
        else
            idx_op = [-i, -j, i, j]
            idx_ψ = collect(-1:-1:(-N))
            idx_ψ[i], idx_ψ[j] = i, j
            ψ2 = ncon([op, ψ], [idx_op, idx_ψ])
            n2[idx] = (ψ' * ψ2).data[1]
        end
    end
    @test n2 ≈ n
end
