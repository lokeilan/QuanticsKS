# Fixed-potential options and results; requested and occupied state counts are separate.

"Fixed-H eigensolver dimensions are independent of the physical occupation count."
Base.@kwdef struct FixedSpectrumOptions
    states::Int = 56
    occupied_states::Int = 50
    max_iterations::Int = 60
    ritz_tolerance::Float64 = 1e-12
    matrix_tolerance::Float64 = 1e-13
    diagnostic_tolerance::Float64 = 1e-13
    virtual_envelope_amplitude::Float64 = 0.5
    virtual_envelope_period::Float64 = 4.0
    virtual_wavevectors::Vector{Int} = [-25, 26, -26, 27, -27, 28]
    seed::Int = 20261031
    method::Symbol = :block_green
    energy_tolerance::Float64 = 1e-6
    min_iterations::Int = 2
    direction_relative_cutoff::Float64 = 1e-6
    direction_absolute_cutoff::Float64 = 1e-10
    direction_residual_floor::Float64 = 1e-8
    use_previous_subspace::Bool = true
    previous_projection_floor::Float64 = 1e-6
    projection::Symbol = :streaming_grid
    diagnostics::Symbol = :streaming_grid
    grid_block_size::Int = 256
    adaptive_work::Bool = true
    max_work_tolerance::Float64 = 1e-6
    work_residual_factor::Float64 = 0.01
    adaptive_ritz::Bool = true
    max_ritz_tolerance::Float64 = 1e-8
    ritz_residual_factor::Float64 = 0.01
    search_basis::Symbol = :direct
    metric_relative_cutoff::Float64 = 1e-12
    metric_absolute_cutoff::Float64 = 1e-24
    green_tolerance_factor::Float64 = 1.0
    preconditioner_form::Symbol = :residual
    direction_norm_floor::Float64 = 1e-12
end
struct FixedSpectrumResult
    problem::Problem
    orbitals::Vector{Q.QTT}
    eigenvalues::Vector{Float64}
    occupied::OrbitalState
    history::Vector{Any}
    converged::Bool
    elapsed_seconds::Float64
end
function validate(o::FixedSpectrumOptions)
    0 < o.occupied_states <= o.states && o.max_iterations > 0 ||
        throw(ArgumentError("Invalid spectrum dimensions"))
    all(
        x -> isfinite(x) && x > 0,
        (o.ritz_tolerance, o.matrix_tolerance, o.diagnostic_tolerance, o.virtual_envelope_period),
    ) || throw(ArgumentError("Invalid fixed spectrum controls"))
    isfinite(o.virtual_envelope_amplitude) || throw(ArgumentError("Invalid virtual envelope"))
    o.method in (:lrrap, :block_residual, :block_green) ||
        throw(ArgumentError("Unknown fixed spectrum method"))
    o.min_iterations >= 1 && isfinite(o.energy_tolerance) && o.energy_tolerance > 0 ||
        throw(ArgumentError("Invalid fixed-H stopping criterion"))
    o.search_basis in (:projected, :direct) || throw(ArgumentError("Unknown search basis"))
    o.preconditioner_form == :residual || throw(
        ArgumentError(
            "Filtered trial-space expansion is unsupported; use method=:block_green for the simple Green iteration",
        ),
    )
    o.projection in (:streaming_grid, :qtt_gradient) &&
        o.diagnostics in (:streaming_grid, :qtt) &&
        o.grid_block_size > 0 || throw(ArgumentError("Invalid projection/diagnostic controls"))
    all(
        x -> isfinite(x) && x > 0,
        (
            o.direction_relative_cutoff,
            o.direction_absolute_cutoff,
            o.direction_residual_floor,
            o.previous_projection_floor,
            o.max_work_tolerance,
            o.work_residual_factor,
            o.max_ritz_tolerance,
            o.ritz_residual_factor,
        ),
    ) || throw(ArgumentError("Invalid residual-subspace controls"))
    all(
        x -> isfinite(x) && x > 0,
        (
            o.metric_relative_cutoff,
            o.metric_absolute_cutoff,
            o.green_tolerance_factor,
            o.direction_norm_floor,
        ),
    ) || throw(ArgumentError("Invalid metric or Green rounding controls"))
    length(o.virtual_wavevectors) == o.states - o.occupied_states ||
        throw(ArgumentError("One analytic wavevector is required for each added virtual state"))
    o
end
