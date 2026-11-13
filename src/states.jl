# Orbital cores store sqrt(dV)*psi, so Euclidean inner products give spatial integrals.

"Orbitals store amplitudes u_j=sqrt(dV)*psi(r_j), as QTTs or dense vectors."
struct OrbitalState{P<:Problem,V}
    problem::P
    orbitals::Vector{V}
    time::Float64
    function OrbitalState(problem::Problem, orbitals::Vector{Q.QTT}; time = 0.0)
        length(orbitals) == length(problem.occupations) ||
            throw(DimensionMismatch("Orbital/occupation count"))
        isfinite(time) || throw(ArgumentError("Invalid time"))
        for psi in orbitals
            length(psi) == nsites(problem.grid) || throw(DimensionMismatch("QTT grid"))
            size(first(psi.c), 1) == size(last(psi.c), 3) == 1 ||
                throw(DimensionMismatch("QTT boundary"))
            all(g -> size(g, 2) == 2 && all(isfinite, g), psi.c) ||
                throw(ArgumentError("Invalid QTT cores"))
            all(k -> size(psi.c[k], 3) == size(psi.c[k + 1], 1), 1:(length(psi) - 1)) ||
                throw(DimensionMismatch("QTT bonds"))
            Q.ttnorm(psi) > 0 || throw(ArgumentError("Zero orbital"))
        end
        new{typeof(problem),Q.QTT}(problem, orbitals, time)
    end
    function OrbitalState(
        problem::Problem,
        orbitals::AbstractVector{<:AbstractVector{<:Number}};
        time = 0.0,
    )
        length(orbitals) == length(problem.occupations) ||
            throw(DimensionMismatch("Orbital/occupation count"))
        isfinite(time) || throw(ArgumentError("Invalid time"))
        all(psi -> length(psi) == length(problem.grid), orbitals) ||
            throw(DimensionMismatch("Dense orbital grid"))
        all(psi -> all(isfinite, psi) && norm(psi) > 0, orbitals) ||
            throw(ArgumentError("Invalid dense orbital"))
        new{typeof(problem),Vector{ComplexF64}}(
            problem,
            [convert(Vector{ComplexF64}, psi) for psi in orbitals],
            time,
        )
    end
end
representation(::OrbitalState{<:Problem,Q.QTT}) = :qtt
representation(::OrbitalState{<:Problem,<:AbstractVector}) = :dense
function state_from_functions(
    problem::Problem,
    functions;
    accuracy = Accuracy(),
    tci = TCIOptions(),
    orthogonalize = true,
    time = 0.0,
    parallel_tasks = 1,
    representation::Symbol = :qtt,
)
    length(functions) == length(problem.occupations) ||
        throw(DimensionMismatch("Function/occupation count"))
    validate(accuracy)
    cfg = solver_config(problem, accuracy, SCFOptions(; parallel_tasks))
    rec = recorder(nothing, tci; parallel_tasks)
    dx = volume_element(problem.grid)
    representation in (:qtt, :dense) ||
        throw(ArgumentError("Unknown orbital representation: $representation"))
    if representation == :dense
        orbitals = [
            [
                ComplexF64(sqrt(dx) * sample_function(f, grid_position(problem.grid, j))) for
                j = 0:(length(problem.grid) - 1)
            ] for f in functions
        ]
        orbitals = normalize.(orbitals)
        orthogonalize && (orbitals = first(Q.orthogonalize(orbitals, cfg, rec)))
        return OrbitalState(problem, orbitals; time)
    end
    orbitals = Q.recorded_map(Q.QTT, length(functions), cfg, rec) do i, localrec
        f = functions[i]
        Q.normalize(
            Q.cross(
                localrec,
                nsites(problem.grid),
                j -> sqrt(dx) * sample_function(f, grid_position(problem.grid, j));
                tol = accuracy.orthogonalization_tolerance,
                cap = accuracy.orbital_cap,
                label = "initial_orbital_$i",
            ),
        )
    end
    orthogonalize && (orbitals = first(Q.orthogonalize(orbitals, cfg, rec)))
    OrbitalState(problem, orbitals; time)
end
initial_state(problem, accuracy, rec, guess::InitialGuess) =
    OrbitalState(problem, initial_orbitals(problem, accuracy, rec, guess))

initial_orbitals(problem, accuracy, rec, guess, ::Val{:qtt}; kwargs...) =
    initial_orbitals(problem, accuracy, rec, guess; kwargs...)

"One periodic envelope and mode order for dense and QTT initialization."
function periodic_initial_samples(problem, guess, count)
    grid = problem.grid
    period = something(
        guess.envelope_period,
        problem.model isa SoftCoulombLSDA1D ? grid.L / length(problem.model.positions) : grid.L,
    )
    isfinite(period) && period > 0 || throw(ArgumentError("Invalid envelope period"))
    wavevectors =
        isempty(guess.wavevectors) ? [iseven(i) ? i ÷ 2 : -(i ÷ 2) for i = 1:count] :
        guess.wavevectors
    length(wavevectors) == count && length(unique(mod.(wavevectors, length(grid)))) == count ||
        throw(ArgumentError("Provide one distinct grid wavevector per orbital"))
    dx, ngrid = spacing(grid), length(grid)
    envelope =
        j -> exp(
            guess.envelope_amplitude * cos(2pi * j * dx / period) +
            guess.slow_cosine_amplitude * cos(2pi * guess.slow_cosine_mode * j / ngrid) +
            guess.slow_sine_amplitude * sin(2pi * guess.slow_sine_mode * j / ngrid),
        )
    envelope, wavevectors
end

function initial_orbitals(
    problem,
    accuracy,
    rec,
    guess,
    ::Val{:dense};
    count = length(problem.occupations),
)
    validate(guess)
    grid = problem.grid
    orbitals = if grid isa PeriodicGrid1D
        envelope, wavevectors = periodic_initial_samples(problem, guess, count)
        amplitudes = normalize(envelope.(0:(length(grid) - 1)))
        [
            ComplexF64[
                amplitudes[j + 1] * cis(2pi * k * j / length(grid)) for j = 0:(length(grid) - 1)
            ] for k in wavevectors
        ]
    else
        [
            normalize(ComplexF64[f(grid_position(grid, j)) for j = 0:(length(grid) - 1)]) for
            f in initial_box_functions(problem, guess, count)
        ]
    end
    cfg = solver_config(
        problem,
        accuracy,
        SCFOptions(parallel_tasks = get(rec.options, :parallel_tasks, 1)),
    )
    first(Q.orthogonalize(orbitals, cfg, rec))
end

function initial_orbitals(
    problem,
    accuracy,
    rec,
    guess::InitialGuess;
    count = length(problem.occupations),
)
    validate(guess)
    cfg = solver_config(
        problem,
        accuracy,
        SCFOptions(parallel_tasks = get(rec.options, :parallel_tasks, 1)),
    )
    g = problem.grid
    envelope_samples, wavevectors = periodic_initial_samples(problem, guess, count)
    envelope = Q.cross(
        rec,
        g.R,
        envelope_samples;
        tol = cfg.orbital_tolerance / 10,
        cap = accuracy.orbital_cap,
        label = "initial_envelope",
    )
    b = count
    # Fourier phases have unit modulus, so every mode shares the envelope norm.
    scales = fill(1 / Q.ttnorm(envelope), b)
    orbitals = Q.independent_map(Q.QTT, b, cfg) do i
        psi = Q.phase_tt(envelope, wavevectors[i])
        Q.scale(psi, scales[i])
    end
    # Every initial orbital has the same envelope. Reconstruct its Fourier sum
    # directly instead of contracting a separate train for every occupied mode.
    combination_samples = initial_orbital_samples(envelope, view(wavevectors, 1:b), scales, rec)
    first(Q.orthogonalize(orbitals, cfg, rec; combination_samples))
end

"Sample normalized Fourier modes of one shared initial envelope."
function initial_orbital_samples(envelope, wavevectors, scales, rec)
    R = length(envelope)
    ngrid = 1 << R
    b = length(wavevectors)
    samples = Q.synchronized(Q.memoized_sample(envelope, rec))
    phases = [cis(2pi * wavevectors[i] * 2^(k - 1) / ngrid) for k = 1:R, i = 1:b]
    coefficients -> Q.sampled(j -> begin
        value = 0.0im
        for i = 1:b
            phase = 1.0 + 0.0im
            for k = 1:R
                isodd(j >> (k - 1)) && (phase *= phases[k, i])
            end
            value += coefficients[i] * scales[i] * phase
        end
        samples(j) * value
    end, R)
end

function initial_box_functions(problem, guess, norbitals)
    option_dict(guess) == option_dict(InitialGuess()) || throw(
        ArgumentError(
            "3D initial guesses use nuclear envelopes; supply an explicit initial state for customization",
        ),
    )
    grid = problem.grid
    centers = problem.model isa AllElectronLDA3D ? problem.model.positions : [(0.0, 0.0, 0.0)]
    # Order box modes by their continuum kinetic energy in the actual box.
    limits = min.(norbitals, grid_shape(grid))
    modes = sort(
        vec(collect(Iterators.product((1:n for n in limits)...)));
        by = t -> sum((t[a] / grid.L[a])^2 for a = 1:3),
    )
    functions = map(modes[1:norbitals]) do mode
        r ->
            sum(exp(-sqrt(sum(abs2, r .- c))) for c in centers) *
            prod(sinpi(mode[a] * (r[a] / grid.L[a] + 0.5)) for a = 1:3)
    end
    functions
end

function initial_orbitals(
    problem::Problem{<:AbstractModel3D},
    accuracy,
    rec,
    guess::InitialGuess;
    count = length(problem.occupations),
)
    validate(guess)
    grid = problem.grid
    centers = problem.model isa AllElectronLDA3D ? problem.model.positions : [(0.0, 0.0, 0.0)]
    norbitals = count
    functions = initial_box_functions(problem, guess, count)
    pivots = [
        grid_index(
            grid,
            ntuple(
                a -> clamp(
                    floor(Int, (r[a] + grid.L[a] / 2) / spacing(grid)[a]),
                    0,
                    grid_shape(grid)[a] - 1,
                ),
                3,
            ),
        ) for r in centers
    ]
    cfg = solver_config(
        problem,
        accuracy,
        SCFOptions(parallel_tasks = get(rec.options, :parallel_tasks, 1)),
    )
    orbitals = Q.recorded_map(Q.QTT, norbitals, cfg, rec) do i, localrec
        Q.normalize(
            Q.cross(
                localrec,
                nsites(grid),
                j -> functions[i](grid_position(grid, j));
                tol = accuracy.orthogonalization_tolerance,
                cap = accuracy.orbital_cap,
                label = "initial_orbital_$i",
                pivots,
            ),
        )
    end
    first(Q.orthogonalize(orbitals, cfg, rec))
end

"Periodic polynomial interpolation, including the new volume-element normalization."
function refinement_samples(psi, coarse::PeriodicGrid1D, grid, refinement, rec)
    Q.PeriodicRefinement(
        psi,
        grid.R;
        degree = refinement.degree,
        max_bytes = rec.options.environment_cache_max_bytes,
    )
end

"Interpolate cell-centered amplitudes with zero exterior values and the new volume element."
function refinement_samples(psi, coarse::BoxGrid3D, grid, refinement, rec)
    ratio = 1 .<< (axis_bits(grid) .- axis_bits(coarse))
    shape = grid_shape(coarse)
    nodes = collect((-(refinement.degree ÷ 2)):(refinement.degree ÷ 2 + 1))
    evaluator = psi isa Q.QTT ? Q.evaluate(psi, rec) : j -> psi[j + 1]
    j -> begin
        index = grid_indices(grid, j)
        coordinate = ntuple(a -> (index[a] + 0.5) / ratio[a] - 0.5, 3)
        base = floor.(Int, coordinate)
        weights = ntuple(
            a -> [
                prod((coordinate[a] - base[a] - b) / (v - b) for b in nodes if b != v)
                for v in nodes
            ],
            3,
        )
        value = 0.0im
        for indices in Iterators.product(eachindex(nodes), eachindex(nodes), eachindex(nodes))
            position = ntuple(a -> base[a] + nodes[indices[a]], 3)
            all(a -> 0 <= position[a] < shape[a], 1:3) || continue
            value +=
                prod(weights[a][indices[a]] for a = 1:3) * evaluator(grid_index(coarse, position))
        end
        value / sqrt(prod(ratio))
    end
end

"Refine QTT or dense orbitals on the same physical domain, then restore orthogonality."
function refine_state(
    state::OrbitalState,
    target::Problem;
    accuracy = Accuracy(),
    tci = TCIOptions(),
    refinement = RefinementOptions(),
    parallel_tasks = 1,
)
    compatible_problem(state.problem, target; refinement = true) ||
        throw(ArgumentError("Refinement changes the physical problem"))
    validate(accuracy)
    validate(refinement)
    coarse, grid = state.problem.grid, target.grid
    added = grid isa PeriodicGrid1D ? (grid.R - coarse.R,) : axis_bits(grid) .- axis_bits(coarse)
    all(r -> 0 <= r <= refinement.max_added_bits, added) && any(added .> 0) ||
        throw(ArgumentError("Invalid refinement increment"))
    rec = recorder(nothing, tci; parallel_tasks)
    cfg = solver_config(target, accuracy, SCFOptions(; parallel_tasks))
    qtt = representation(state) == :qtt
    orbitals = Q.recorded_map(
        eltype(state.orbitals),
        length(state.orbitals),
        cfg,
        rec;
        random = qtt,
    ) do i, localrec
        samples = refinement_samples(state.orbitals[i], coarse, grid, refinement, localrec)
        if qtt
            Q.cross(
                localrec,
                nsites(grid),
                samples;
                tol = min(refinement.tolerance, cfg.orbital_tolerance),
                cap = accuracy.orbital_cap,
                label = "refine_$i",
                require_convergence = accuracy.adaptive,
            )
        else
            ComplexF64[samples(j) for j = 0:(length(grid) - 1)]
        end
    end
    orbitals, gram_defect = Q.orthogonalize(orbitals, cfg, rec)
    if qtt
        orbitals = first(
            Q.compress_orbitals(
                orbitals,
                cfg;
                tol = accuracy.storage_tolerance,
                input_gram_defect = gram_defect,
            ),
        )
    end
    OrbitalState(target, orbitals; time = state.time)
end

"Rows are orbitals; columns are internal bonds ordered from MSB to LSB."
bond_profiles(s::OrbitalState) = reduce(
    vcat,
    [permutedims(reverse([size(g, 3) for g in psi.c[1:(end - 1)]])) for psi in s.orbitals],
)
bond_profiles(s::OrbitalState{<:Problem,<:AbstractVector}) = zeros(Int, length(s.orbitals), 0)

# Save orbital cores and problem metadata; check model and grid identity on load.

const CHECKPOINT_SCHEMA = 1
"Write a versioned data-only Julia checkpoint. Load only trusted local files."
function save_checkpoint(path, state::OrbitalState; metadata = Dict())
    mkpath(dirname(abspath(path)))
    data = Dict(
        "schema_version" => CHECKPOINT_SCHEMA,
        "package_version" => "0.1.0",
        "julia_version" => string(VERSION),
        "problem" => problem_descriptor(state.problem),
        "time" => state.time,
        "representation" => string(representation(state)),
        "metadata" => metadata,
    )
    if representation(state) == :qtt
        data["cores"] = [x.c for x in state.orbitals]
    else
        data["orbitals"] = state.orbitals
    end
    # Only built-in container/scalar/array types are serialized; no module-owned types or closures.
    tmp = path * ".tmp"
    open(tmp, "w") do io
        serialize(io, data)
    end
    mv(tmp, path; force = true)
    path
end
function restore_problem(d; model = nothing)
    get(d, "dimensions", 1) == 3 && return restore_problem3d(d; model)
    d["boundary"] == "periodic" && d["units"] == "atomic" ||
        throw(ArgumentError("Unsupported checkpoint conventions"))
    md = d["model"]
    if md["kind"] == "soft_coulomb_lsda_1d"
        m = SoftCoulombLSDA1D(
            md["positions"],
            md["charges"];
            softening = md["softening"],
            coupling = md["coupling"],
            xc = Symbol(md["xc"]),
        )
        model !== nothing && model_descriptor(model) != md && throw(ArgumentError("Model mismatch"))
    elseif md["kind"] == "local_potential_1d"
        model === nothing && throw(
            ArgumentError(
                "Supply the original LocalPotential1D model; executable functions are not serialized",
            ),
        )
        model_descriptor(model) == md || throw(ArgumentError("Potential id mismatch"))
        m = model
    else
        throw(ArgumentError("Unknown model kind"))
    end
    Problem(PeriodicGrid1D(d["R"], d["cell"]), m, d["occupations"]; mass = d["mass"])
end
function restore_problem3d(d; model = nothing)
    d["boundary"] == "isolated_zero_exterior" &&
        d["units"] == "atomic" &&
        d["ordering"] == "interleaved_lsb" &&
        d["nodes"] == "cell_centers" ||
        throw(ArgumentError("Unsupported 3D checkpoint conventions"))
    md = d["model"]
    m = if md["kind"] == "all_electron_lda_3d"
        AllElectronLDA3D(md["positions"], md["charges"]; xc = Symbol(md["xc"]))
    elseif md["kind"] == "local_potential_3d" && model isa LocalPotential3D
        model
    else
        throw(ArgumentError("Supply the original LocalPotential3D for custom-potential checkpoints"))
    end
    model !== nothing && model_descriptor(model) != md && throw(ArgumentError("Model mismatch"))
    Problem(BoxGrid3D(d["R"], d["cell"]), m, d["occupations"]; mass = d["mass"])
end

function load_checkpoint(path; model = nothing, problem = nothing)
    d = deserialize(path)
    d isa AbstractDict && get(d, "schema_version", 0) == CHECKPOINT_SCHEMA ||
        throw(ArgumentError("Unsupported checkpoint schema; expected schema $CHECKPOINT_SCHEMA"))
    restored = restore_problem(d["problem"]; model = problem === nothing ? model : problem.model)
    problem !== nothing &&
        !compatible_problem(problem, restored) &&
        throw(ArgumentError("Checkpoint problem mismatch"))
    kind = Symbol(get(d, "representation", "qtt"))
    kind in (:qtt, :dense) || throw(ArgumentError("Unknown checkpoint representation: $kind"))
    orbitals = kind == :qtt ? [Q.QTT(c) for c in d["cores"]] : d["orbitals"]
    OrbitalState(restored, orbitals; time = d["time"])
end

# Keep spectral smoothing separate from the Hamiltonian and solver workflows.

struct LowPassMPO
    grid::PeriodicGrid1D
    mass::Float64
    cores::Vector{Array{ComplexF64,4}}
end

"""
    lowpass_mpo(grid; cutoff, mass = 1)
    lowpass_mpo(problem; cutoff)

Construct a reusable periodic QTT low-pass MPO. `cutoff` is the FD2 kinetic
energy in Hartree at which the amplitude gain is one half.
"""
function lowpass_mpo(grid::PeriodicGrid1D; cutoff::Real, mass::Real = 1)
    m = Float64(mass)
    isfinite(m) && m > 0 || throw(ArgumentError("Low-pass mass must be finite and positive"))
    cores = Q.lowpass_mpo(grid.R, spacing(grid) * sqrt(m), Float64(cutoff))
    LowPassMPO(grid, m, cores)
end
lowpass_mpo(::BoxGrid3D; kwargs...) =
    throw(ArgumentError("The low-pass interface is currently 1D only"))
lowpass_mpo(problem::Problem; cutoff) = lowpass_mpo(problem.grid; cutoff, mass = problem.mass)

"""
    apply_lowpass(filter, state; tol = 1e-12, cap = 64, parallel_tasks = 1)

Filter each orbital on the original grid. The returned amplitudes retain
the filter's attenuation. Normalize or reorthogonalize separately if needed.
"""
function apply_lowpass(
    filter::LowPassMPO,
    state::OrbitalState;
    tol = 1e-12,
    cap = 64,
    parallel_tasks = 1,
)
    problem = state.problem
    filter.grid == problem.grid && filter.mass == problem.mass ||
        throw(ArgumentError("Low-pass filter requires the same grid and mass as the state"))
    orbitals = Q.independent_map(Q.QTT, length(state.orbitals), (; parallel_tasks)) do i
        Q.apply_mpo(filter.cores, state.orbitals[i]; tol, cap)
    end
    OrbitalState(problem, orbitals; time = state.time)
end
