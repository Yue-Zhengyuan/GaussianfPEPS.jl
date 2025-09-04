using Random
using LinearAlgebra
using TensorKit
using PEPSKit
using GaussianfPEPS
import TensorKitTensors.HubbardOperators as HO

Random.seed!(32178046)

# χ is the number of virtual fermions along each direction
function test(χ::Int = 2)
    # number of physical fermions at each site
    Np = 2
    U, G, = GaussianfPEPS.generate_cormat(Np, χ)
    peps = translate(U, Np, χ)
    lattice = collect(space(t, 1) for t in peps.A)

    Espace = Vect[FermionParity](0 => 4, 1 => 4)
    env = CTMRGEnv(randn, ComplexF64, peps, Espace)
    for χenv in [8, 16]
        trscheme = truncdim(χenv) & truncerr(1.0e-12)
        env, = leading_boundary(
            env, peps; tol = 1.0e-10, maxiter = 100, trscheme,
            alg = :sequential, projector_alg = :fullinfinite
        )
    end

    bz = BrillouinZone((128, 128), (false, true))

    O = LocalOperator(lattice, ((1, 1),) => HO.e_num(Trivial, Trivial))
    doping1 = 1 - real(expectation_value(peps, O, env))
    doping2 = BCS.doping_peps(G, bz, Np)
    @info "Doping" doping1 doping2

    mags1 = map([HO.S_x, HO.S_y, HO.S_z]) do func
        O = LocalOperator(lattice, ((1, 1),) => func(Trivial, Trivial))
        real(expectation_value(peps, O, env))
    end
    mags2 = BCS.mags_peps(G, bz, Np)
    @info "Magnetization" mags1 mags2

    singlets1 = map([(1, 2), (2, 1)]) do site2
        O = LocalOperator(lattice, ((1, 1), site2) => -HO.singlet_min(Trivial, Trivial))
        expectation_value(peps, O, env)
    end
    singlets2 = map([[1, 0], [0, -1]]) do v
        BCS.singlet_peps(G, bz, Np, v)
    end
    @info "NN singlet pairing" singlets1 singlets2

    hoppings1 = map([(1, 2), (2, 1)]) do site2
        O = LocalOperator(lattice, ((1, 1), site2) => HO.e_hopping(Trivial, Trivial))
        real(expectation_value(peps, O, env))
    end
    hoppings2 = map([[1, 0], [0, -1]]) do v
        BCS.hopping_peps(G, bz, Np, v)
    end
    @info "NN hopping energy" hoppings1 hoppings2

    t, Δx, Δy, mu = 1.0, 0.3, -0.7, -0.4
    ham = BCS.hamiltonian(ComplexF64, InfiniteSquare(1, 1); t, Δx, Δy, mu)
    energy1 = expectation_value(peps, ham, env)
    energy2 = BCS.energy_peps(G, bz, Np; Δx, Δy, t, mu)
    @info "PEPS energy per site" energy1 energy2

    return nothing
end

test(2)
