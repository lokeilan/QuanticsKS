# Load fixed-potential options, diagnostics, updates, and the solver loop.

# Fixed-potential eigenvalue problem. These files share the QuanticsKS module.
include("spectrum/options.jl")
include("spectrum/diagnostics.jl")
include("spectrum/updates.jl")
include("spectrum/solve.jl")
