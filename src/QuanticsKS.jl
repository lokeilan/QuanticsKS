# Load QTT kernels first, followed by models, states, and solver workflows.

module QuanticsKS

using LinearAlgebra, Random, Serialization, JSON3, Dates, Printf, SHA, TOML

module Numerics
using LinearAlgebra, Random, Serialization, TensorOperations
import TensorCrossInterpolation as TCI
import ITensors, ITensorMPS

include("numerics/svd_safety.jl")
include("numerics/parallel.jl")
include("numerics/tensor.jl")
include("numerics/sampling.jl")
include("numerics/tci.jl")
include("numerics/physics.jl")
include("numerics/hamiltonian.jl")
include("numerics/lrrap.jl")
end
const Q = Numerics

include("models.jl")
include("models3d.jl")
include("states.jl")
include("hamiltonian.jl")
include("scf.jl")
include("dynamics.jl")
include("observables.jl")
include("spectrum.jl")
include("configuration.jl")
include("three_dimensional.jl")

export PeriodicGrid1D, SoftCoulombLSDA1D, LocalPotential1D, Problem, hydrogen_chain
export Accuracy, SCFOptions, TimeOptions, OrbitalState, SCFResult, TimeResult
export TCIOptions, InitialGuess, RefinementOptions, AuditOptions, PhysicsNumerics
export state_from_functions, ground_state, refine_state, propagate, observe, bond_profiles
export lowpass_mpo, apply_lowpass
export save_checkpoint, load_checkpoint, postprocess, run_config
export configuration_plan
export NoDrive, CosinePulse, FunctionDrive
export FixedSpectrumOptions, FixedSpectrumResult, fixed_spectrum
export BoxGrid3D, AllElectronLDA3D, LocalPotential3D, hydrogen_cluster
export grid_index, grid_indices, grid_position, nsites, volume_element

end
