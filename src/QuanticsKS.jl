# Load QTT kernels first, followed by models, states, and solver workflows.

module QuanticsKS

using LinearAlgebra, Random, Serialization, JSON3, Dates, Printf, SHA, TOML

module Numerics
using LinearAlgebra, Random, Serialization, TensorOperations
import TensorCrossInterpolation as TCI
import ITensors, ITensorMPS
import KrylovKit
import LOBPCGEigensolver
import VectorInterface as VI

include("tensor.jl")
include("tci.jl")
include("physics.jl")
include("solvers.jl")
end
const Q = Numerics
using .Numerics: block_lobpcg

include("models.jl")
include("states.jl")
include("hamiltonian.jl")
include("scf.jl")
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
export block_lobpcg
export BoxGrid3D, AllElectronLDA3D, LocalPotential3D
export grid_index, grid_indices, grid_position, nsites, volume_element

end
