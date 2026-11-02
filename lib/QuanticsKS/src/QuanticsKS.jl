# Load QTT kernels first, followed by models, states, and solver workflows.

module QuanticsKS

using LinearAlgebra, Random, Serialization, JSON3, Dates, Printf, SHA, TOML

module Numerics
using LinearAlgebra, Random, Serialization
include("numerics/tt.jl")
include("numerics/instrumentation.jl")
include("numerics/physics.jl")
include("numerics/lrrap.jl")
include("numerics/interpolation.jl")
include("numerics/iteration_support.jl")
end
const Q = Numerics

include("systems.jl")
include("states.jl")
include("hamiltonian.jl")
include("checkpoints.jl")
include("runs.jl")
include("scf.jl")
include("dynamics.jl")
include("observables.jl")
include("configuration.jl")
include("fixed_spectrum.jl")

export PeriodicGrid1D, SoftCoulombLSDA1D, LocalPotential1D, Problem, hydrogen_chain
export Accuracy, SCFOptions, TimeOptions, OrbitalState, SCFResult, TimeResult
export TCIOptions, InitialGuess, RefinementOptions, AuditOptions, PhysicsNumerics
export state_from_functions, ground_state, refine_state, propagate, observe, bond_profiles
export save_checkpoint, load_checkpoint, postprocess, run_config
export NoDrive, CosinePulse, FunctionDrive
export FixedSpectrumOptions, FixedSpectrumResult, fixed_spectrum

end
