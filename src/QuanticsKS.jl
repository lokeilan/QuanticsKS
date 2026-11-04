# Load QTT kernels first, followed by models, states, and solver workflows.

module QuanticsKS

using LinearAlgebra, Random, Serialization, JSON3, Dates, Printf, SHA, TOML

include("numerics.jl")
const Q = Numerics

include("models.jl")
include("options.jl")
include("drives.jl")
include("states.jl")
include("hamiltonian.jl")
include("checkpoints.jl")
include("runs.jl")
include("scf_output_ritz.jl")
include("scf.jl")
include("scf_audit.jl")
include("dynamics.jl")
include("observables.jl")
include("configuration.jl")
include("spectrum/options.jl")
include("spectrum/diagnostics.jl")
include("spectrum/updates.jl")
include("spectrum/solve.jl")
include("cli.jl")

export PeriodicGrid1D, SoftCoulombLSDA1D, LocalPotential1D, Problem, hydrogen_chain
export Accuracy, SCFOptions, TimeOptions, OrbitalState, SCFResult, TimeResult
export TCIOptions, InitialGuess, RefinementOptions, AuditOptions, PhysicsNumerics
export state_from_functions, ground_state, refine_state, propagate, observe, bond_profiles
export save_checkpoint, load_checkpoint, postprocess, run_config
export configuration_plan
export NoDrive, CosinePulse, FunctionDrive
export FixedSpectrumOptions, FixedSpectrumResult, fixed_spectrum

end
