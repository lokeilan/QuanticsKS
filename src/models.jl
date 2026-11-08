# Periodic 1D grids, models, and occupations; Problem uses atomic units.

abstract type AbstractGrid end
abstract type AbstractModel end
"Uniform periodic grid x_j=j*L/2^R, with atomic units throughout."
struct PeriodicGrid1D <: AbstractGrid
    R::Int
    L::Float64
    function PeriodicGrid1D(R::Integer, L::Real)
        2 <= R <= 40 || throw(ArgumentError("R must be between 2 and 40"))
        isfinite(L) && L > 0 || throw(ArgumentError("Cell length must be finite and positive"))
        new(R, L)
    end
end
Base.length(g::PeriodicGrid1D) = 1 << g.R
spacing(g::PeriodicGrid1D) = g.L / length(g)
Base.:(==)(a::PeriodicGrid1D, b::PeriodicGrid1D) = a.R == b.R && a.L == b.L

abstract type AbstractModel1D <: AbstractModel end
"Periodic a=1 soft Coulomb and unpolarized 1D LSDA; fixed nuclei."
struct SoftCoulombLSDA1D <: AbstractModel1D
    positions::Vector{Float64}
    charges::Vector{Float64}
    softening::Float64
    coupling::Float64
    xc::Symbol
    function SoftCoulombLSDA1D(
        positions,
        charges = ones(length(positions));
        softening = 1.0,
        coupling = 1.0,
        xc = :lsda,
    )
        !isempty(positions) && length(positions) == length(charges) ||
            throw(ArgumentError("Nuclear arrays must have equal positive length"))
        all(isfinite, positions) && all(z -> isfinite(z) && z > 0, charges) ||
            throw(ArgumentError("Invalid nuclei"))
        isfinite(softening) && softening > 0 && isfinite(coupling) && coupling > 0 ||
            throw(ArgumentError("Invalid interaction parameters"))
        xc in (:lsda, :none) || throw(ArgumentError("XC must be :lsda or :none"))
        xc == :lsda &&
            (softening != 1 || coupling != 1) &&
            throw(
                ArgumentError(
                    "This LSDA parametrization is valid only for softening=coupling=1; use xc=:none for Hartree experiments",
                ),
            )
        new(collect(positions), collect(charges), softening, coupling, xc)
    end
end
"Noninteracting particles in a real periodic potential f(x); id identifies its physical definition."
struct LocalPotential1D{F} <: AbstractModel1D
    potential::F
    id::String
    function LocalPotential1D(f; id::AbstractString)
        isempty(id) && throw(ArgumentError("A stable potential id is required for checkpoints"))
        new{typeof(f)}(f, id)
    end
end
struct ConstantPotential
    value::Float64
end
(f::ConstantPotential)(x) = f.value
function LocalPotential1D(value::Real; id = "constant:$value")
    isfinite(value) || throw(ArgumentError("Constant potential must be finite"))
    LocalPotential1D(ConstantPotential(value); id)
end
nsites(g::PeriodicGrid1D) = g.R
cell_volume(g::PeriodicGrid1D) = g.L
volume_element(g::PeriodicGrid1D) = spacing(g)
grid_position(g::PeriodicGrid1D, j) = j * spacing(g)
sample_function(f, x::Real) = f(x)
sample_function(f, x::Tuple) = f(x...)
validate_model_grid(g::AbstractGrid, m::AbstractModel, w, mass) =
    throw(ArgumentError("Incompatible grid and physical model"))
validate_model_grid(g::PeriodicGrid1D, m::AbstractModel1D, w, mass) = nothing

struct Problem{M<:AbstractModel,G<:AbstractGrid}
    grid::G
    model::M
    occupations::Vector{Float64}
    mass::Float64
    function Problem(
        grid::G,
        model::M,
        occupations;
        mass = 1.0,
    ) where {M<:AbstractModel,G<:AbstractGrid}
        isfinite(mass) && mass > 0 || throw(ArgumentError("Mass must be positive"))
        w = Float64.(collect(occupations))
        0 < length(w) <= length(grid) && all(x -> isfinite(x) && x > 0, w) ||
            throw(ArgumentError("Invalid occupations"))
        validate_model_grid(grid, model, w, mass)
        if model isa SoftCoulombLSDA1D
            model.softening <= grid.L / 4 || throw(
                ArgumentError("The current 28-term periodic kernel requires softening <= cell/4"),
            )
            model.xc == :lsda &&
                mass != 1 &&
                throw(ArgumentError("This LSDA parametrization assumes electron mass 1"))
            all(==(2.0), w) || throw(
                ArgumentError(
                    "Restricted unpolarized LSDA currently requires occupation 2 for every orbital",
                ),
            )
            isapprox(sum(w), sum(model.charges); atol = 1e-12, rtol = 0) ||
                throw(ArgumentError("Periodic soft-Coulomb systems must be neutral"))
            positions = mod.(model.positions, grid.L)
            length(unique(positions)) == length(positions) ||
                throw(ArgumentError("Coincident nuclei"))
        end
        new{M,G}(grid, model, w, mass)
    end
end
function hydrogen_chain(;
    R = 16,
    natoms = 100,
    distance = 4.0,
    cell = natoms * distance,
    offset = 0.0,
    softening = 1.0,
    coupling = 1.0,
    xc = :lsda,
    mass = 1.0,
)
    natoms > 0 && iseven(natoms) || throw(ArgumentError("Use a positive even number of H atoms"))
    Problem(
        PeriodicGrid1D(R, cell),
        SoftCoulombLSDA1D(offset .+ distance .* collect(0:(natoms - 1)); softening, coupling, xc),
        fill(2.0, natoms ÷ 2);
        mass,
    )
end

model_descriptor(m::SoftCoulombLSDA1D) = Dict(
    "kind" => "soft_coulomb_lsda_1d",
    "positions" => m.positions,
    "charges" => m.charges,
    "softening" => m.softening,
    "coupling" => m.coupling,
    "xc" => string(m.xc),
)
model_descriptor(m::LocalPotential1D) = Dict("kind" => "local_potential_1d", "id" => m.id)
function problem_descriptor(p)
    Dict(
        "R" => p.grid.R,
        "cell" => p.grid.L,
        "occupations" => p.occupations,
        "mass" => p.mass,
        "model" => model_descriptor(p.model),
        "boundary" => "periodic",
        "units" => "atomic",
    )
end
function compatible_problem(a, b; refinement = false)
    typeof(a.grid) == typeof(b.grid) && a.grid.L == b.grid.L &&
        a.mass == b.mass &&
        (refinement || a.grid.R == b.grid.R) &&
        a.occupations == b.occupations &&
        model_descriptor(a.model) == model_descriptor(b.model)
end

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
    # Unset component caps inherit the legacy field budget.
    external_cap::Union{Nothing,Int} = nothing
    kernel_cap::Union{Nothing,Int} = nothing
    density_cap::Union{Nothing,Int} = nothing
    hartree_cap::Union{Nothing,Int} = nothing
    effective_potential_cap::Union{Nothing,Int} = nothing
    xc_energy_cap::Union{Nothing,Int} = nothing
    drive_cap::Union{Nothing,Int} = nothing
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
"Resolve independent field budgets without changing legacy inputs."
field_caps(a::Accuracy) = (
    external_cap = something(a.external_cap, a.field_cap),
    kernel_cap = something(a.kernel_cap, a.field_cap),
    density_cap = something(a.density_cap, a.field_cap),
    hartree_cap = something(a.hartree_cap, a.field_cap),
    effective_potential_cap = something(a.effective_potential_cap, a.field_cap),
    xc_energy_cap = something(a.xc_energy_cap, a.field_cap),
    drive_cap = something(a.drive_cap, a.field_cap),
)
Base.@kwdef struct TCIOptions
    max_iterations::Int = 16
    check_history::Int = 2
    global_pivot_search::Int = 5
    max_global_pivots::Int = 5
    initial_random_pivots::Int = 6
    initial_pivots::Vector{Int} = Int[]
    # Zero keeps the prrLU result; a positive value explicitly requests SVD rounding.
    post_rounding_factor::Float64 = 0.0
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
    green_relative_tolerance::Float64 = 1e-8
    green_max_iterations::Int = 300
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
    for (name, cap) in pairs(field_caps(a))
        cap > 0 || throw(ArgumentError("$name must be positive"))
    end
    a
end
function solver_config(p, a, s = SCFOptions())
    validate(a)
    merge(
        (;
            R = nsites(p.grid),
            cell = cell_volume(p.grid),
            volume_element = volume_element(p.grid),
            electrons = sum(p.occupations),
            rankcap = a.orbital_cap,
            opcap = field_caps(a).effective_potential_cap,
            density_cap = field_caps(a).density_cap,
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

# Time-dependent local scalar potentials and their validation.

abstract type AbstractDrive end
struct NoDrive <: AbstractDrive end
"Periodic scalar pulse with a sin² envelope; amplitude is a potential energy in Ha."
Base.@kwdef struct CosinePulse <: AbstractDrive
    amplitude::Float64 = 0.01
    start::Float64 = 0.0
    duration::Float64 = 1.0
    frequency::Float64 = 0.0
    phase::Float64 = 0.0
    spatial_mode::Int = 1
    spatial_phase::Float64 = 0.0
end
struct FunctionDrive{F} <: AbstractDrive
    potential::F
    id::String
end
drive_value(::NoDrive, x, t, L) = 0.0
function drive_value(d::CosinePulse, x, t, L)
    tau = t - d.start
    0 < tau < d.duration || return 0.0
    d.amplitude *
    sinpi(tau / d.duration)^2 *
    cos(d.frequency * tau + d.phase) *
    cos(2pi * d.spatial_mode * x / L + d.spatial_phase)
end
function drive_value(d::FunctionDrive, x, t, L)
    v = d.potential(x, t)
    isreal(v) && isfinite(v) ||
        throw(ArgumentError("Drive must return finite real potential energies"))
    real(v)
end
drive_descriptor(::NoDrive) = Dict("kind" => "none")
drive_descriptor(d::CosinePulse) = merge(Dict("kind" => "cosine_pulse"), option_dict(d))
drive_descriptor(d::FunctionDrive) = Dict("kind" => "function", "id" => d.id)
function validate(d::AbstractDrive)
    if d isa CosinePulse
        d.duration > 0 && all(
            isfinite,
            (d.amplitude, d.start, d.duration, d.frequency, d.phase, d.spatial_phase),
        ) || throw(ArgumentError("Invalid pulse"))
    elseif d isa FunctionDrive
        isempty(d.id) && throw(ArgumentError("Drive id is required"))
    end
    d
end
