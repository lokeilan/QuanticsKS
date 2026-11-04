# Load tensor kernels before sampling, physical fields, and orbital updates.
module Numerics

using LinearAlgebra, Random, Serialization, TensorOperations
import TensorCrossInterpolation as TCI

include("numerics/svd_safety.jl")
include("numerics/parallel.jl")
include("numerics/tensor.jl")
include("numerics/sample_cache.jl")
include("numerics/evaluation.jl")
include("numerics/linear_expressions.jl")
include("numerics/sampled_ops.jl")
include("numerics/mpo.jl")
include("numerics/tangent.jl")
include("numerics/cross_state.jl")
include("numerics/instrumentation.jl")
include("numerics/cross.jl")
include("numerics/recorded_parallel.jl")
include("numerics/physics.jl")
include("numerics/hamiltonian.jl")
include("numerics/orbital_basis.jl")
include("numerics/coefficients.jl")
include("numerics/lrrap.jl")
include("numerics/interpolation.jl")
include("numerics/iteration_support.jl")

end
