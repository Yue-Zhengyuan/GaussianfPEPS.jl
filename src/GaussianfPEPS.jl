module GaussianfPEPS

export BrillouinZone
export fiducial_cormat, cormat_to_nx, state_to_nx
export virtual_state, fiducial_state, get_peps
export parent_Hamiltonian_BdG, bogoliubov, bogoliubov_blocks
export parent_Hamiltonian, parent_Hamiltonian_map
export load_orth, save_tensor

using LinearAlgebra
# using Manifolds, Manopt
using SparseArrays: sparse, blockdiag
using HDF5, JLD2
using TensorKit
using PEPSKit
import TensorKitTensors.FermionOperators as FO

const V = FO.fermion_space()
const unit = TensorKit.id(V)
const iσy = sparse([0.0 1.0; -1.0 0.0])
const w = sparse([0.5 0.5im])
const w2 = sparse([1.0 1.0; 1.0im -1.0im])

include("bz.jl")
include("cor_mat.jl")
# include("optimize.jl")
include("states.jl")
include("bogoliubov.jl")
include("load_py.jl")
include("io.jl")

end # module GaussianfPEPS
