# Load the 3D periodic model; joint and FCC states share the outer SCF loop.

module SiliconJointStep2

using LinearAlgebra, TOML, QuanticsKS, Random, Serialization, JSON3, Printf, Mmap, Dates
const Q = QuanticsKS.Q

export JointLayout,
    JointState,
    encode,
    decode,
    kpoint,
    sample_joint,
    gram_at,
    density_at,
    rotate_joint,
    kinetic_at,
    storage_report,
    read_plan,
    build_physics,
    dense_scf,
    joint_scf,
    measure_state,
    native_reference,
    compare_dense,
    FCCState

include("representations/joint_grid.jl")
include("representations/joint_state.jl")
include("input/joint_plan.jl")
include("models/pseudopotential_3d.jl")
include("input/pseudopotential_file.jl")
include("operators/bloch_fd8.jl")
include("solvers/joint_eigen.jl")
include("observables/joint.jl")
include("scf/periodic_mixing.jl")
include("scf/dense_k.jl")
include("io/dense_reference.jl")
include("input/joint_initial_state.jl")
include("scf/joint.jl")
include("representations/fcc_state.jl")
include("input/fcc_initial_state.jl")
include("operators/fcc.jl")
include("solvers/fcc_eigen.jl")
include("observables/fcc.jl")

end
