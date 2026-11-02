# Orbital cores store u_j=sqrt(h)*psi(x_j), so Euclidean inner products give spatial integrals.

"QTT amplitudes u_j=sqrt(h)*psi(x_j); no full-grid orbital array is stored."
struct OrbitalState{P<:Problem}
    problem::P
    orbitals::Vector{Q.QTT}
    time::Float64
    function OrbitalState(p::Problem, xs::Vector{Q.QTT}; time = 0.0)
        length(xs) == length(p.occupations) || throw(DimensionMismatch("Orbital/occupation count"))
        isfinite(time) || throw(ArgumentError("Invalid time"))
        for x in xs
            length(x) == p.grid.R || throw(DimensionMismatch("QTT grid"))
            size(first(x.c), 1) == size(last(x.c), 3) == 1 ||
                throw(DimensionMismatch("QTT boundary"))
            all(g -> size(g, 2) == 2 && all(isfinite, g), x.c) ||
                throw(ArgumentError("Invalid QTT cores"))
            all(k -> size(x.c[k], 3) == size(x.c[k + 1], 1), 1:(length(x) - 1)) ||
                throw(DimensionMismatch("QTT bonds"))
            Q.ttnorm(x) > 0 || throw(ArgumentError("Zero orbital"))
        end
        new{typeof(p)}(p, xs, time)
    end
end
function state_from_functions(
    p::Problem,
    functions;
    accuracy = Accuracy(),
    tci = TCIOptions(),
    orthogonalize = true,
    time = 0.0,
)
    length(functions) == length(p.occupations) ||
        throw(DimensionMismatch("Function/occupation count"))
    a = validate(accuracy)
    cfg = solver_config(p, a)
    rec = recorder(nothing, tci)
    h = spacing(p.grid)
    xs = [
        Q.normalize(
            Q.cross(
                rec,
                p.grid.R,
                j -> sqrt(h) * f(j * h);
                tol = a.orthogonalization_tolerance,
                cap = a.orbital_cap,
                label = "initial_orbital",
            ),
        ) for f in functions
    ]
    orthogonalize && (xs = first(Q.orthogonalize(xs, cfg, rec)))
    OrbitalState(p, xs; time)
end
function initial_state(p, a, rec, guess::InitialGuess)
    validate(guess)
    cfg = solver_config(p, a)
    g = p.grid
    h = spacing(g)
    N = length(g)
    scale = something(
        guess.envelope_period,
        p.model isa SoftCoulombLSDA1D ? g.L / length(p.model.positions) : g.L,
    )
    isfinite(scale) && scale > 0 || throw(ArgumentError("Invalid envelope period"))
    envelope = Q.cross(
        rec,
        g.R,
        j -> exp(
            guess.envelope_amplitude * cos(2pi * j * h / scale) +
            guess.slow_cosine_amplitude * cos(2pi * guess.slow_cosine_mode * j / N) +
            guess.slow_sine_amplitude * sin(2pi * guess.slow_sine_mode * j / N),
        );
        tol = a.field_tolerance / 10,
        cap = a.orbital_cap,
        label = "initial_envelope",
    )
    ks = Int[0]
    b = length(p.occupations)
    for k = 1:b
        push!(ks, k)
        push!(ks, -k)
    end
    if !isempty(guess.wavevectors)
        length(guess.wavevectors) == b && length(unique(mod.(guess.wavevectors, N))) == b ||
            throw(ArgumentError("Provide one distinct grid wavevector per orbital"))
        ks = guess.wavevectors
    end
    xs = [Q.normalize(Q.phase_tt(envelope, k)) for k in ks[1:b]]
    OrbitalState(p, first(Q.orthogonalize(xs, cfg, rec)))
end
"Periodic six-point degree-five refinement; creates a warm start, not an SCF-history continuation."
function refine_state(
    state::OrbitalState,
    target::Problem;
    accuracy = Accuracy(),
    tci = TCIOptions(),
    refinement = RefinementOptions(),
)
    compatible_problem(state.problem, target; refinement = true) ||
        throw(ArgumentError("Refinement changes the physical problem"))
    target.grid.R > state.problem.grid.R || throw(ArgumentError("Target must be finer"))
    validate(refinement)
    target.grid.R - state.problem.grid.R <= refinement.max_added_bits ||
        throw(ArgumentError("Refinement exceeds configured bit increment"))
    a = validate(accuracy)
    rec = recorder(nothing, tci)
    cfg = solver_config(target, a)
    xs = [
        Q.cross(
            rec,
            target.grid.R,
            Q.PeriodicRefinement(
                x,
                target.grid.R;
                degree = refinement.degree,
                cachebits = tci.environment_cache_bits,
            );
            tol = refinement.tolerance,
            cap = a.orbital_cap,
            label = "refine_$i",
        ) for (i, x) in enumerate(state.orbitals)
    ]
    xs = first(Q.orthogonalize(xs, cfg, rec))
    xs = first(Q.pack_orbitals(xs, cfg; tol = a.storage_tolerance))
    OrbitalState(target, xs; time = state.time)
end
"Rows are orbitals; columns are internal bonds ordered from MSB to LSB."
bond_profiles(s::OrbitalState) =
    reduce(vcat, [permutedims(reverse([size(g, 3) for g in x.c[1:(end - 1)]])) for x in s.orbitals])
