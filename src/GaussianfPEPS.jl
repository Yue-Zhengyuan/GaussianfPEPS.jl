module GaussianfPEPS

export BrillouinZone
export fiducial_cormat, cormat_to_nx, state_to_nx
export paired_state, virtual_state, fiducial_state, get_peps
export parent_Hamiltonian_BdG, bogoliubov, bogoliubov_blocks
export bilinear_Hamiltonian, bilinear_Hamiltonian_map
export translate

export BCS

using LinearAlgebra
using SparseArrays: sparse, blockdiag, spdiagm
using HDF5
using TensorKit
using PEPSKit
import TensorKitTensors.FermionOperators as FO

const V = FO.fermion_space()
const unit = TensorKit.id(V)
const iσy = sparse([0.0 1.0; -1.0 0.0])
const w = sparse([0.5 0.5im])
const w2 = sparse([1.0 1.0; 1.0im -1.0im])

include("bz.jl")
include("cormat.jl")
include("states.jl")
include("bogoliubov.jl")
include("utils.jl")

include("models/bcs.jl")

end # module GaussianfPEPS
