# Load the model, operators, eigensolvers, and SCF for the periodic 1D dense reference.

module H100Reference

using LinearAlgebra, Random, Dates, TOML, Serialization, Printf
using FFTW, QuadGK, SpecialFunctions, SparseArrays, Arpack

include("QTTReferencePhysics.jl")
include("input/dense_1d.jl")
include("scf/dense_mixing.jl")
include("io/dense_output.jl")
include("operators/fd2.jl")
include("operators/fft_1d.jl")
include("models/soft_coulomb_1d.jl")
include("solvers/block_eigen.jl")
include("solvers/cyclic_eigen.jl")
include("scf/dense_block.jl")
include("scf/dense_cyclic.jl")

end
