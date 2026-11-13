# Load QTT kernels first, followed by models, states, and solver workflows.

module QuanticsKS

using LinearAlgebra, Random, Serialization, JSON3, Dates, Printf, TOML
using SparseArrays, FFTW
import Libdl
import KrylovKit

module Numerics
using LinearAlgebra, Random, Serialization, Dates, JSON3, TensorOperations
import TensorCrossInterpolation as TCI
import ITensors, ITensorMPS
import KrylovKit
import VectorInterface as VI

include("tensor.jl")
include("tci.jl")
include("physics.jl")
include("solvers.jl")
end
const Q = Numerics

include("models.jl")
include("correlated.jl")
include("xc.jl")
include("states.jl")
include("hamiltonian.jl")
include("scf.jl")
include("propagation_operators.jl")
include("dynamics.jl")
include("observables.jl")
include("spectrum.jl")
include("configuration.jl")

export PeriodicGrid1D, SoftCoulombLSDA1D, LocalPotential1D, Problem, hydrogen_chain
export Accuracy, SCFOptions, TimeOptions, OrbitalState, SCFResult, TimeResult
export TCIOptions, InitialGuess, RefinementOptions, AuditOptions, PhysicsNumerics
export state_from_functions, ground_state, refine_state, propagate, observe, bond_profiles
export lowpass_mpo, apply_lowpass
export save_checkpoint, load_checkpoint, postprocess, run_config
export configuration_plan
export NoDrive, CosinePulse, FunctionDrive
export FixedSpectrumOptions, FixedSpectrumResult, fixed_spectrum
export BoxGrid3D, AllElectronLDA3D, LocalPotential3D
export grid_index, grid_indices, grid_position, nsites, volume_element
export LineGrid1D, SoftCoulomb1D, CorrelatedProblem, CorrelatedState
export correlated_eigenstate, correlated_observe, correlated_propagate, kick
export compress_orbitals, compress_state
export save_correlated_checkpoint, load_correlated_checkpoint

end
