# Orbital cores store sqrt(dV)*psi, so Euclidean inner products give spatial integrals.

"QTT amplitudes u_j=sqrt(dV)*psi(r_j); no full-grid orbital array is stored."
struct OrbitalState{P<:Problem}
    problem::P
    orbitals::Vector{Q.QTT}
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
        new{typeof(problem)}(problem, orbitals, time)
    end
end
function state_from_functions(
    problem::Problem,
    functions;
    accuracy = Accuracy(),
    tci = TCIOptions(),
    orthogonalize = true,
    time = 0.0,
    parallel_tasks = 1,
)
    length(functions) == length(problem.occupations) ||
        throw(DimensionMismatch("Function/occupation count"))
    validate(accuracy)
    cfg = solver_config(problem, accuracy, SCFOptions(; parallel_tasks))
    rec = recorder(nothing, tci; parallel_tasks)
    dx = volume_element(problem.grid)
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
function initial_state(problem, accuracy, rec, guess::InitialGuess)
    validate(guess)
    cfg = solver_config(
        problem,
        accuracy,
        SCFOptions(parallel_tasks = get(rec.options, :parallel_tasks, 1)),
    )
    g = problem.grid
    dx = spacing(g)
    ngrid = length(g)
    scale = something(
        guess.envelope_period,
        problem.model isa SoftCoulombLSDA1D ? g.L / length(problem.model.positions) : g.L,
    )
    isfinite(scale) && scale > 0 || throw(ArgumentError("Invalid envelope period"))
    envelope = Q.cross(
        rec,
        g.R,
        j -> exp(
            guess.envelope_amplitude * cos(2pi * j * dx / scale) +
            guess.slow_cosine_amplitude * cos(2pi * guess.slow_cosine_mode * j / ngrid) +
            guess.slow_sine_amplitude * sin(2pi * guess.slow_sine_mode * j / ngrid),
        );
        tol = accuracy.field_tolerance / 10,
        cap = accuracy.orbital_cap,
        label = "initial_envelope",
    )
    wavevectors = Int[0]
    b = length(problem.occupations)
    for k = 1:b
        push!(wavevectors, k)
        push!(wavevectors, -k)
    end
    if !isempty(guess.wavevectors)
        length(guess.wavevectors) == b && length(unique(mod.(guess.wavevectors, ngrid))) == b ||
            throw(ArgumentError("Provide one distinct grid wavevector per orbital"))
        wavevectors = guess.wavevectors
    end
    orbitals =
        Q.independent_map(i -> Q.normalize(Q.phase_tt(envelope, wavevectors[i])), Q.QTT, b, cfg)
    OrbitalState(problem, first(Q.orthogonalize(orbitals, cfg, rec)))
end

function initial_state(p::Problem{<:AbstractModel3D}, a, rec, guess::InitialGuess)
    validate(guess)
    option_dict(guess) == option_dict(InitialGuess()) || throw(
        ArgumentError(
            "3D initial guesses use nuclear envelopes; supply an explicit initial state for customization",
        ),
    )
    g = p.grid
    centers = p.model isa AllElectronLDA3D ? p.model.positions : [(0.0, 0.0, 0.0)]
    b = length(p.occupations)
    mode_limit = ceil(Int, cbrt(b))
    modes = sort(
        vec(collect(Iterators.product(1:mode_limit, 1:mode_limit, 1:mode_limit)));
        by = t -> sum(abs2, t),
    )
    cfg = solver_config(p, a)
    orbitals = Q.QTT[]
    for i = 1:b
        mode = modes[i]
        fn = j -> begin
            r = grid_position(g, j)
            envelope = sum(exp(-sqrt(sum(abs2, r .- c))) for c in centers)
            envelope * prod(sinpi(mode[k] * (r[k] / g.L[k] + 0.5)) for k = 1:3)
        end
        push!(
            orbitals,
            Q.normalize(
                Q.cross(
                    rec,
                    nsites(g),
                    fn;
                    tol = a.orthogonalization_tolerance,
                    cap = a.orbital_cap,
                    label = "initial_orbital_$i",
                ),
            ),
        )
    end
    OrbitalState(p, first(Q.orthogonalize(orbitals, cfg, rec)))
end

"Periodic six-point degree-five refinement; creates a warm start, not an SCF-history continuation."
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
    target.grid.R > state.problem.grid.R || throw(ArgumentError("Target must be finer"))
    validate(refinement)
    target.grid.R - state.problem.grid.R <= refinement.max_added_bits ||
        throw(ArgumentError("Refinement exceeds configured bit increment"))
    validate(accuracy)
    rec = recorder(nothing, tci; parallel_tasks)
    cfg = solver_config(target, accuracy, SCFOptions(; parallel_tasks))
    orbitals = Q.recorded_map(Q.QTT, length(state.orbitals), cfg, rec) do i, localrec
        psi = state.orbitals[i]
        Q.cross(
            localrec,
            target.grid.R,
            Q.PeriodicRefinement(
                psi,
                target.grid.R;
                degree = refinement.degree,
                cachebits = tci.environment_cache_bits,
            );
            tol = refinement.tolerance,
            cap = accuracy.orbital_cap,
            label = "refine_$i",
        )
    end
    orbitals = first(Q.orthogonalize(orbitals, cfg, rec))
    orbitals = first(Q.pack_orbitals(orbitals, cfg; tol = accuracy.storage_tolerance))
    OrbitalState(target, orbitals; time = state.time)
end

function refine_state(
    state::OrbitalState{<:Problem{<:AbstractModel3D}},
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
    coarse = state.problem.grid
    g = target.grid
    0 < g.R - coarse.R <= refinement.max_added_bits ||
        throw(ArgumentError("Invalid 3D refinement increment"))
    rec = recorder(nothing, tci; parallel_tasks)
    cfg = solver_config(target, accuracy, SCFOptions(; parallel_tasks))
    ratio = 1 << (g.R - coarse.R)
    n = 1 << coarse.R
    nodes = collect((-(refinement.degree ÷ 2)):(refinement.degree ÷ 2 + 1))
    orbitals = Q.recorded_map(Q.QTT, length(state.orbitals), cfg, rec) do i, localrec
        ev = Q.evaluate(state.orbitals[i], localrec)
        fn =
            j -> begin
                q = grid_indices(g, j)
                u = ntuple(a -> (q[a] + 0.5) / ratio - 0.5, 3)
                base = floor.(Int, u)
                weights = ntuple(
                    a -> [
                        prod((u[a] - base[a] - b) / (v - b) for b in nodes if b != v) for v in nodes
                    ],
                    3,
                )
                value = 0.0im
                for ks in Iterators.product(eachindex(nodes), eachindex(nodes), eachindex(nodes))
                    index = ntuple(a -> base[a] + nodes[ks[a]], 3)
                    all(t -> 0 <= t < n, index) || continue
                    value += prod(weights[a][ks[a]] for a = 1:3) * ev(grid_index(coarse, index))
                end
                value / sqrt(ratio^3)
            end
        Q.cross(
            localrec,
            nsites(g),
            fn;
            tol = refinement.tolerance,
            cap = accuracy.orbital_cap,
            label = "refine_$i",
        )
    end
    orbitals = first(Q.orthogonalize(orbitals, cfg, rec))
    orbitals = first(Q.pack_orbitals(orbitals, cfg; tol = accuracy.storage_tolerance))
    OrbitalState(target, orbitals; time = state.time)
end

"Rows are orbitals; columns are internal bonds ordered from MSB to LSB."
bond_profiles(s::OrbitalState) = reduce(
    vcat,
    [permutedims(reverse([size(g, 3) for g in psi.c[1:(end - 1)]])) for psi in s.orbitals],
)

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
        "cores" => [x.c for x in state.orbitals],
        "metadata" => metadata,
    )
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
    OrbitalState(restored, [Q.QTT(c) for c in d["cores"]]; time = d["time"])
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
