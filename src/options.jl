# Numerical accuracy and iteration options; physical parameters live in models.jl.

# field, work, and storage set error budgets for fields, iteration tensors, and saved orbitals.
Base.@kwdef struct Accuracy
    field_tolerance::Float64 = 1e-8
    work_tolerance::Float64 = 1e-6
    storage_tolerance::Float64 = 1e-8
    algebra_tolerance::Float64 = 1e-12
    # Optional SCF sampling budgets; nothing retains the algebra/work defaults.
    mixing_tolerance::Union{Nothing,Float64} = nothing
    density_residual_tolerance::Union{Nothing,Float64} = nothing
    green_rhs_cap::Union{Nothing,Int} = nothing
    orbital_cap::Int = 48
    field_cap::Int = 64
    work_cap::Int = 96
    density_tolerance::Float64 = 1e-10
    kernel_tolerance::Float64 = 1e-12
    external_tolerance::Float64 = 1e-12
    xc_energy_tolerance::Float64 = 1e-10
    orthogonalization_tolerance::Float64 = 1e-11
    green_rhs_tolerance::Float64 = 1e-9
    pack_gram_tolerance::Float64 = 1e-7
    pack_attempts::Int = 4
end
Base.@kwdef struct TCIOptions
    max_iterations::Int = 16
    check_history::Int = 2
    global_pivot_search::Int = 5
    max_global_pivots::Int = 5
    initial_random_pivots::Int = 6
    initial_pivots::Vector{Int} = Int[]
    # Post-fit SVD tolerance is this factor times the TCI tolerance.
    post_rounding_factor::Float64 = 0.1
    scalar_cache_capacity::Int = 16384
    dense_cache_max_points::Int = 262144
    orbital_cache_capacity::Int = 8192
    orbital_cache_max_bytes::Int = 64 * 1024^2
    slice_cache_capacity::Int = 2048
    reuse_pivots::Bool = true
    pivot_cache_capacity::Int = 512
    orbital_pivot_group_size::Int = 4
    environment_cache_bits::Int = 10
    require_convergence::Bool = false
    zero_check_limit::Int = 4096
end
Base.@kwdef struct InitialGuess
    envelope_amplitude::Float64 = 0.5
    envelope_period::Union{Nothing,Float64} = nothing
    slow_cosine_amplitude::Float64 = 0.005
    slow_sine_amplitude::Float64 = 0.003
    slow_cosine_mode::Int = 1
    slow_sine_mode::Int = 3
    wavevectors::Vector{Int} = Int[]
end
Base.@kwdef struct RefinementOptions
    degree::Int = 5
    tolerance::Float64 = 1e-11
    max_added_bits::Int = 8
end
# Audit settings provide separate rank limits and tolerances for the final state checks.
Base.@kwdef struct AuditOptions
    rank_multiplier::Int = 2
    field_tolerance::Float64 = 1e-9
    kernel_tolerance::Float64 = 1e-13
    algebra_tolerance::Float64 = 1e-13
    gram_tolerance::Float64 = 1e-6
    charge_tolerance::Float64 = 1e-6
end
Base.@kwdef struct PhysicsNumerics
    xc_table_points::Int = 2048
    xc_density_min::Float64 = 1e-8
    xc_density_max::Float64 = 4.0
    exchange_quadrature_rtol::Float64 = 2e-11
end
Base.@kwdef struct SCFOptions
    # 0 uses the default Julia thread pool; 1 keeps core contractions serial.
    parallel_tasks::Int = 1
    energy_tolerance::Float64 = 1e-6
    max_iterations::Int = 100
    inner_steps::Int = 1
    coefficient_sweeps::Int = 2
    mixing::Float64 = 0.3
    anderson::Bool = true
    green_shift::Float64 = 0.5
    eigen_tolerance::Float64 = 1e-4
    density_tolerance::Float64 = 1e-4
    criterion::Symbol = :energy
    seed::Int = 20261030
    anderson_history::Int = 5
    anderson_regularization::Float64 = 1e-6
    anderson_coefficient_limit::Float64 = 5.0
    green_polish_every::Int = 5
    green_polish_residual::Float64 = 0.1
    tangent_relative_cutoff::Float64 = 1e-7
    tangent_absolute_cutoff::Float64 = 1e-12
    metric_relative_cutoff::Float64 = 1e-11
    metric_absolute_cutoff::Float64 = 1e-13
    constraint_relative_cutoff::Float64 = 1e-10
    constraint_absolute_cutoff::Float64 = 1e-12
    orthogonalize_threshold::Float64 = 1e-8
    independence_threshold::Float64 = 1e-8
    gram_tolerance::Float64 = 1e-6
end
function validate(a::Accuracy)
    all(
        x -> isfinite(x) && x > 0,
        (
            a.field_tolerance,
            a.work_tolerance,
            a.algebra_tolerance,
            a.density_tolerance,
            a.kernel_tolerance,
            a.external_tolerance,
            a.xc_energy_tolerance,
            a.orthogonalization_tolerance,
            a.green_rhs_tolerance,
            a.pack_gram_tolerance,
        ),
    ) || throw(ArgumentError("Tolerances must be positive and finite"))
    for tol in (a.mixing_tolerance, a.density_residual_tolerance)
        tol === nothing ||
            (isfinite(tol) && tol > 0) ||
            throw(ArgumentError("SCF sampling tolerances must be positive and finite"))
    end
    a.green_rhs_cap === nothing ||
        a.green_rhs_cap > 0 ||
        throw(ArgumentError("Green RHS rank cap must be positive"))
    isfinite(a.storage_tolerance) && a.storage_tolerance >= 0 ||
        throw(ArgumentError("Invalid storage tolerance"))
    min(a.orbital_cap, a.field_cap, a.work_cap, a.pack_attempts) > 0 ||
        throw(ArgumentError("Bond caps and attempts must be positive"))
    a
end
function solver_config(p, a, s = SCFOptions())
    validate(a)
    merge(
        (;
            R = p.grid.R,
            cell = p.grid.L,
            electrons = sum(p.occupations),
            rankcap = a.orbital_cap,
            opcap = a.field_cap,
            workcap = a.work_cap,
            tci_tol = a.field_tolerance,
            round_tol = a.algebra_tolerance,
            eigen_tol = s.eigen_tolerance,
            inner_steps = s.inner_steps,
            coefficient_sweeps = s.coefficient_sweeps,
            green_shift = s.green_shift,
            mixing = s.mixing,
            anderson = s.anderson,
        ),
        (; (k => getfield(s, k) for k in fieldnames(SCFOptions))...),
        (;
            ortho_tol = a.orthogonalization_tolerance,
            rhs_tol = a.green_rhs_tolerance,
            rhs_cap = something(a.green_rhs_cap, a.work_cap),
            mixing_tol = something(a.mixing_tolerance, a.algebra_tolerance),
            density_residual_tol = something(a.density_residual_tolerance, a.algebra_tolerance),
            pack_gram_tolerance = a.pack_gram_tolerance,
            pack_attempts = a.pack_attempts,
        ),
    )
end

Base.@kwdef struct TimeOptions
    parallel_tasks::Int = 1
    dt::Float64 = 0.01
    steps::Int = 100
    snapshot_every::Int = 10
    observe_every::Int = 1
    phase_tolerance::Float64 = 1e-10
    phase_cap::Int = 48
    kinetic_tolerance::Float64 = 1e-12
    kinetic_cap::Int = 96
    output_tolerance::Float64 = 1e-12
    output_cap::Int = 48
    norm_drift_limit::Float64 = 1e-5
    gram_drift_limit::Float64 = 1e-5
    seed::Int = 20261031
end
function validate(t::TimeOptions)
    t.parallel_tasks >= 0 || throw(ArgumentError("parallel_tasks must be nonnegative"))
    all(
        x -> isfinite(x) && x > 0,
        (
            t.dt,
            t.phase_tolerance,
            t.kinetic_tolerance,
            t.output_tolerance,
            t.norm_drift_limit,
            t.gram_drift_limit,
        ),
    ) || throw(ArgumentError("Invalid propagation tolerances/time step"))
    min(t.steps, t.snapshot_every, t.observe_every, t.phase_cap, t.kinetic_cap, t.output_cap) > 0 ||
        throw(ArgumentError("Invalid propagation counts/caps"))
    t
end
