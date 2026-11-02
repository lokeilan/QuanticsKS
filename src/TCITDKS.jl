# Shared interface for input validation, size estimates, and configured solvers.

# Keep the original module identities: Julia checkpoints record these names.
# In the command-line entry point they remain Main.H100Reference and
# Main.SiliconJointStep2, so old checkpoints and postprocessing scripts work.
if !isdefined(@__MODULE__, :H100Reference)
    include("H100Reference.jl")
end
if !isdefined(@__MODULE__, :SiliconJointStep2)
    include("SiliconJointStep2.jl")
end

module TCITDKS

using Dates, LinearAlgebra, Serialization, SHA, TOML, JSON3, FFTW
import QuanticsKS

import ..H100Reference
import ..SiliconJointStep2

export read_input, calculation_plan, run_calculation

# Physics is selected while reading the input. Numerical methods receive a
# prepared calculation and do not inspect the name of the chemical system.
include("input/calculation.jl")
include("input/hydrogen.jl")
include("input/silicon.jl")
include("io/run_files.jl")
include("workflow.jl")

end
