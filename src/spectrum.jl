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
    o.method in (:lrrap, :block_residual, :block_green, SUBSPACE_SOLVERS...) ||
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

# Fixed-H Ritz matrices and residuals using QTT or chunked grid integration.

"Exact finite-grid quadrature in bounded blocks; never stores an N-by-state orbital array."
function spectrum_grid_matrices(orbitals, H, potential, cfg, rec, o; coefficients = nothing)
    ngrid = 2^cfg.R
    b = coefficients === nothing ? length(orbitals) : size(coefficients, 2)
    es = [Q.evaluate(x, rec) for x in orbitals]
    A = zeros(ComplexF64, b, b)
    S = zeros(ComplexF64, b, b)
    for start = 0:(o.grid_block_size):(ngrid - 1)
        count = min(o.grid_block_size, ngrid - start)
        B = Q.independent_columns(count + 1, length(orbitals), cfg) do i
            [es[i](mod(start + j, ngrid)) for j = 0:count]
        end
        coefficients === nothing || (B = B * coefficients)
        X = view(B, 1:count, :)
        D = view(B, 2:(count + 1), :) - X
        S .+= X' * X
        A .+= X' * (view(potential, (start + 1):(start + count)) .* X) + (D' * D) / (2H.h^2)
    end
    A, S
end
function spectrum_grid_residuals(orbitals, H, potential, cfg, rec, o)
    ngrid = 2^cfg.R
    b = length(orbitals)
    es = [Q.evaluate(x, rec) for x in orbitals]
    energies = zeros(b)
    norms = zeros(b)
    for start = 0:(o.grid_block_size):(ngrid - 1)
        count = min(o.grid_block_size, ngrid - start)
        B = Q.independent_columns(count + 1, b, cfg) do i
            [es[i](mod(start + j, ngrid)) for j = 0:count]
        end
        X = view(B, 1:count, :)
        D = view(B, 2:(count + 1), :) - X
        norms .+= vec(sum(abs2, X; dims = 1))
        energies .+=
            vec(sum(view(potential, (start + 1):(start + count)) .* abs2.(X); dims = 1)) +
            vec(sum(abs2, D; dims = 1)) / (2H.h^2)
    end
    energies ./= norms
    r2 = zeros(b)
    for start = 0:(o.grid_block_size):(ngrid - 1)
        count = min(o.grid_block_size, ngrid - start)
        B = Q.independent_columns(count + 2, b, cfg) do i
            [es[i](mod(start + j, ngrid)) for j = -1:count]
        end
        X = view(B, 2:(count + 1), :)
        r =
            (2X - view(B, 1:count, :) - view(B, 3:(count + 2), :)) / (2H.h^2) +
            (view(potential, (start + 1):(start + count)) .- transpose(energies)) .* X
        r2 .+= vec(sum(abs2, r; dims = 1))
    end
    energies, sqrt.(r2 ./ norms)
end

# The fixed-H reference potential may be more accurate than H.potential.
# Fit that shared scalar field once for core contractions, rather than fitting
# one V*x per orbital or silently switching to the looser operator potential.
function spectrum_projection_hamiltonian(H, potential, cfg, rec; tol)
    v =
        all(==(first(potential)), potential) ? Q.constant_tt(cfg.R, first(potential)) :
        Q.cross(
            rec,
            cfg.R,
            j -> potential[j + 1];
            tol,
            cap = cfg.effective_potential_cap,
            label = "fixed_projection_potential",
            pivots = [argmax(j -> abs(potential[j]), eachindex(potential)) - 1],
        )
    Q.Hamiltonian(v, Q.diagonal(v), H.plus, H.minus, H.h)
end

"Project directly from existing cores; reconstruct only the final Ritz orbitals."
function spectrum_ritz(
    orbitals,
    H,
    cfg,
    rec,
    o,
    potential;
    keep = length(orbitals),
    tol = o.ritz_tolerance,
    remove_dependent_directions = false,
)
    if o.projection == :streaming_grid
        A, S = Q.timed(rec, "ritz_projection") do
            spectrum_grid_matrices(orbitals, H, potential, cfg, rec, o)
        end
    else
        A, S = Q.timed(rec, "ritz_projection") do
            projected_H = spectrum_projection_hamiltonian(
                H,
                potential,
                cfg,
                rec;
                tol = o.matrix_tolerance,
            )
            Q.h_gram(orbitals, projected_H; cfg), Q.gram(orbitals; cfg)
        end
    end
    if remove_dependent_directions
        # Preserve every current Ritz vector. Dropping eigenvectors of the raw
        # trial Gram can rotate away part of X and destroy monotonic convergence.
        XZ = view(S, 1:keep, (keep + 1):size(S, 2))
        projection = Hermitian(S[1:keep, 1:keep]) \ XZ
        projector = vcat(
            -projection,
            Matrix{ComplexF64}(I, length(orbitals) - keep, length(orbitals) - keep),
        )
        schur = if o.projection == :streaming_grid
            last(spectrum_grid_matrices(orbitals, H, potential, cfg, rec, o; coefficients = projector))
        else
            S[(keep + 1):end, (keep + 1):end] - XZ' * projection
        end
        W = Q.canonical_orthogonalizer(
            schur,
            o.metric_relative_cutoff,
            max(o.direction_norm_floor^2, o.metric_absolute_cutoff),
        )
        C = zeros(ComplexF64, length(orbitals), keep + size(W, 2))
        C[1:keep, 1:keep] = I(keep)
        C[1:keep, (keep + 1):end] = -projection * W
        C[(keep + 1):end, (keep + 1):end] = W
        reduced, reduced_metric = if o.projection == :streaming_grid
            # Form projected values before quadrature: C'AC loses digits when
            # history directions nearly coincide with the accepted subspace.
            spectrum_grid_matrices(orbitals, H, potential, cfg, rec, o; coefficients = C)
        else
            C' * A * C, C' * S * C
        end
        fac = eigen(
            Hermitian((reduced + reduced') / 2),
            Hermitian((reduced_metric + reduced_metric') / 2),
        )
        coefficients = C * fac.vectors
    else
        fac = Q.timed(rec, "ritz_eigensolve") do
            eigen(Hermitian((A + A') / 2), Hermitian((S + S') / 2))
        end
        coefficients = fac.vectors
    end
    updated_orbitals = Q.timed(rec, "ritz_reconstruction") do
        Q.combine_orbitals(
            orbitals,
            view(coefficients, :, 1:keep),
            cfg,
            rec;
            tol,
            cap = cfg.orbital_cap,
            label = "fixed_ritz",
        )
    end
    updated_orbitals, fac.values[1:keep]
end
function spectrum_residuals(orbitals, H, cfg, o, rec, potential)
    o.diagnostics == :streaming_grid &&
        return spectrum_grid_residuals(orbitals, H, potential, cfg, rec, o)
    projected_H =
        spectrum_projection_hamiltonian(H, potential, cfg, rec; tol = o.diagnostic_tolerance)
    measured = Q.recorded_map(Tuple{Float64,Float64}, length(orbitals), cfg, rec) do i, localrec
        x = orbitals[i]
        e = Q.h_expectation(projected_H, x) / real(Q.inner(x, x))
        residual =
            Q.ttnorm(
                Q.h_residual(
                    projected_H,
                    x,
                    e,
                    cfg,
                    localrec;
                    tol = o.diagnostic_tolerance,
                    cap = cfg.work_cap,
                    label = "fixed_diagnostic_residual_$i",
                    potential = j -> potential[j + 1],
                ),
            ) / Q.ttnorm(x)
        e, residual
    end
    first.(measured), last.(measured)
end

# Green/residual subspace updates at fixed H, with diagnostics for the updated orbitals.

"One Green update per orbital followed by one Rayleigh–Ritz in the updated block."
function simple_green_step(orbitals, energies, H, green, cfg, rec, o, potential, accuracy)
    rec.stage = "block_green"
    updated_orbitals = Q.recorded_map(Q.QTT, length(orbitals), cfg, rec) do i, localrec
        x = orbitals[i]
        rhs = Q.timed(localrec, "green_rhs"; item = string(i)) do
            ex = Q.evaluate(x, localrec)
            Q.cross(
                localrec,
                cfg.R,
                j -> (cfg.green_shift + energies[i] - potential[j + 1]) * ex(j);
                tol = accuracy.green_rhs_tolerance,
                cap = accuracy.work_cap,
                label = "simple_green_rhs_$i",
            )
        end
        y = Q.timed(localrec, "green_mpo"; item = string(i)) do
            Q.apply_mpo(green, rhs; tol = accuracy.algebra_tolerance, cap = accuracy.orbital_cap)
        end
        Q.timed(localrec, "orbital_normalization"; item = string(i)) do
            Q.ttnorm(y) > 1e-14 || error("Green update produced a zero orbital")
            Q.normalize(y)
        end
    end
    rec.stage = "rayleigh_ritz"
    updated_orbitals, eps =
        spectrum_ritz(updated_orbitals, H, cfg, rec, o, potential; tol = o.ritz_tolerance)
    updated_orbitals,
    (
        trial_dimension = length(orbitals),
        rhs_tolerance = accuracy.green_rhs_tolerance,
        green_tolerance = accuracy.algebra_tolerance,
        ritz_tolerance = o.ritz_tolerance,
        search_basis = "updated orbitals only",
    )
end

function spectrum_accepted(o, iteration, energy_change)
    iteration >= o.min_iterations &&
        energy_change !== nothing &&
        isfinite(energy_change) &&
        0 <= energy_change < o.energy_tolerance
end

"Green-preconditioned residual expansion; all trial and accepted vectors remain QTT."
function residual_subspace_step(
    orbitals,
    H,
    green,
    cfg,
    rec,
    o,
    potential;
    work_tolerance,
    ritz_tolerance = o.ritz_tolerance,
    previous = nothing,
    energies = nothing,
    diagnostic_residuals = nothing,
)
    rec.stage = "residual_actions"
    projected_H = spectrum_projection_hamiltonian(H, potential, cfg, rec; tol = work_tolerance)
    candidates = Q.recorded_map(Union{Nothing,Q.QTT}, length(orbitals), cfg, rec) do i, localrec
        x = orbitals[i]
        e = Q.h_expectation(projected_H, x) / real(Q.inner(x, x))
        r = Q.h_residual(
            projected_H,
            x,
            e,
            cfg,
            localrec;
            tol = work_tolerance,
            cap = cfg.work_cap,
            label = "fixed_search_residual_$i",
            potential = j -> potential[j + 1],
        )
        # Search-direction filtering only; this is not a convergence criterion.
        Q.ttnorm(r) > o.direction_residual_floor || return nothing
        # Green removes high-frequency action error; its own truncation must not
        # reintroduce it at the much looser action interpolation tolerance.
        green_tol =
            o.search_basis == :direct ?
            min(work_tolerance, o.green_tolerance_factor * ritz_tolerance) : work_tolerance
        z = Q.apply_mpo(green, r; tol = green_tol, cap = cfg.work_cap)
        Q.ttnorm(z) > 1e-14 ? Q.normalize(z) : nothing
    end
    zs = Q.QTT[z for z in candidates if z !== nothing]
    residual_count = length(zs)
    o.use_previous_subspace && previous !== nothing && append!(zs, previous)
    isempty(zs) && return orbitals, (directions = 0, trial_dimension = length(orbitals))
    if o.search_basis == :direct
        # Equivalent subspace without fitting projected, orthonormal directions.
        # Resolve linear dependence in the small overlap matrix at Rayleigh–Ritz.
        trial = vcat(orbitals, zs)
        rec.stage = "rayleigh_ritz"
        updated_orbitals, eps = spectrum_ritz(
            trial,
            H,
            cfg,
            rec,
            o,
            potential;
            keep = o.states,
            tol = ritz_tolerance,
            remove_dependent_directions = true,
        )
        return updated_orbitals,
        (
            directions = residual_count,
            trial_dimension = length(trial),
            work_tolerance,
            ritz_tolerance,
            green_tolerance = min(work_tolerance, o.green_tolerance_factor * ritz_tolerance),
            search_basis = "direct",
        )
    end
    raw_overlaps = Q.gram(orbitals, zs; cfg)
    overlaps = Hermitian(Q.gram(orbitals; cfg)) \ raw_overlaps
    orbital_evaluator = Q.synchronized(Q.block_evaluator(orbitals, rec))
    rec.stage = "residual_projection"
    candidates = Q.recorded_map(Union{Nothing,Q.QTT}, length(zs), cfg, rec) do i, localrec
        z = zs[i]
        if i > residual_count
            remaining = sqrt(
                max(
                    0,
                    real(Q.inner(z, z)) - real(dot(view(raw_overlaps, :, i), view(overlaps, :, i))),
                ),
            )
            remaining < o.previous_projection_floor && return nothing
        end
        direction_evaluator = Q.evaluate(z, localrec)
        q = Q.linear_combination(
            localrec,
            vcat([z], orbitals),
            vcat(1, -view(overlaps, :, i));
            evaluator = j -> vcat(direction_evaluator(j), orbital_evaluator(j)),
            tol = work_tolerance,
            cap = cfg.work_cap,
            label = "residual_projection_$i",
        )
        Q.ttnorm(q) > 1e-12 ? Q.normalize(q) : nothing
    end
    projected = Q.QTT[z for z in candidates if z !== nothing]
    isempty(projected) && return orbitals, (directions = 0, trial_dimension = length(orbitals))
    S = Q.gram(projected; cfg)
    f = eigen(Hermitian((S + S') / 2))
    good = findall(
        >(max(maximum(f.values) * o.direction_relative_cutoff, o.direction_absolute_cutoff)),
        f.values,
    )
    C = f.vectors[:, good] * Diagonal(1 ./ sqrt.(f.values[good]))
    evaluator = Q.synchronized(Q.block_evaluator(projected, rec))
    rec.stage = "residual_basis"
    directions = Q.recorded_map(Q.QTT, size(C, 2), cfg, rec) do i, localrec
        Q.linear_combination(
            localrec,
            projected,
            view(C, :, i);
            evaluator,
            tol = work_tolerance,
            cap = cfg.work_cap,
            label = "residual_basis_$i",
        )
    end
    trial = vcat(orbitals, directions)
    rec.stage = "rayleigh_ritz"
    updated_orbitals, eps =
        spectrum_ritz(trial, H, cfg, rec, o, potential; keep = o.states, tol = ritz_tolerance)
    updated_orbitals,
    (
        directions = length(directions),
        trial_dimension = length(trial),
        trial_gram_defect = norm(Q.gram(trial; cfg) - I),
        work_tolerance,
        ritz_tolerance,
    )
end

# Fixed-potential iteration and checkpoints; stop on the configured occupied-energy change.

function spectrum_checkpoint(dir, problem, orbitals, eps, history, o, potential_hash)
    dir === nothing && return
    data = Dict(
        "schema_version" => 1,
        "kind" => "fixed_spectrum",
        "problem" => problem_descriptor(problem),
        "cores" => [x.c for x in orbitals],
        "eigenvalues" => eps,
        "occupied_states" => o.occupied_states,
        "history" => history,
        "potential_sha256" => potential_hash,
    )
    tmp = joinpath(dir, "spectrum.jls.tmp")
    serialize(tmp, data)
    mv(tmp, joinpath(dir, "spectrum.jls"); force = true)
    save_checkpoint(
        joinpath(dir, "occupied-checkpoint.jls"),
        OrbitalState(problem, orbitals[1:(o.occupied_states)]);
        metadata = Dict(
            "fixed_hamiltonian" => true,
            "solved_states" => o.states,
            "selection" => "lowest occupied_states after Rayleigh-Ritz",
        ),
    )
end

"Optimize QTT eigenstates of a supplied fixed potential; only selected states enter the physical density."
function fixed_spectrum(
    initial::OrbitalState,
    potential::AbstractVector{<:Real};
    spectrum = FixedSpectrumOptions(),
    accuracy = Accuracy(),
    solver = nothing,
    scf = SCFOptions(green_update_interval = 1),
    tci = TCIOptions(),
    output = nothing,
    resume_from = nothing,
)
    if solver isa SCFOptions
        # Preserve the original keyword for callers supplying numerical controls.
        scf = solver
    elseif solver isa Symbol
        spectrum = FixedSpectrumOptions(; merge(option_tuple(spectrum), (; method = solver))...)
    elseif solver !== nothing
        throw(ArgumentError("solver must be a method Symbol or SCFOptions"))
    end
    o = validate(spectrum)
    validate(accuracy)
    validate(scf)
    validate(tci)
    use_subspace = o.method in SUBSPACE_SOLVERS
    criterion = use_subspace ? "residual_and_gram" : "energy"
    problem = initial.problem
    grid = problem.grid
    ngrid = length(grid)
    dx = spacing(grid)
    length(potential) == ngrid && all(isfinite, potential) ||
        throw(ArgumentError("Invalid fixed potential"))
    length(initial.orbitals) == length(problem.occupations) == o.occupied_states ||
        throw(ArgumentError("Physical occupations must describe only the selected occupied states"))
    o.states <= ngrid || throw(ArgumentError("More states than grid points"))
    potential_hash = bytes2hex(sha256(reinterpret(UInt8, Float64.(potential))))
    resumed = nothing
    if resume_from !== nothing
        data = deserialize(resume_from)
        get(data, "kind", nothing) == "fixed_spectrum" && get(data, "schema_version", 0) == 1 ||
            throw(ArgumentError("Not a supported spectrum checkpoint"))
        data["problem"] == problem_descriptor(problem) &&
            data["occupied_states"] == o.occupied_states &&
            length(data["cores"]) == o.states ||
            throw(ArgumentError("Resume problem or state count mismatch"))
        hash = get(data, "potential_sha256", nothing)
        if hash === nothing
            sidecar = joinpath(dirname(resume_from), "metadata.json")
            isfile(sidecar) ||
                throw(ArgumentError("Legacy spectrum requires its potential-hash metadata sidecar"))
            hash = JSON3.read(read(sidecar, String))["potential_sha256"]
        end
        hash == potential_hash || throw(ArgumentError("Resume fixed potential mismatch"))
        resumed = [Q.QTT(c) for c in data["cores"]]
    end
    t0 = time_ns()
    Random.seed!(o.seed)
    dir = prepare_run(
        output,
        "fixed_spectrum",
        problem;
        spectrum = o,
        accuracy = accuracy,
        solver = scf,
        tci,
    )
    rec = recorder(dir, tci; parallel_tasks = scf.parallel_tasks)
    io = run_io(dir, "iterations.jsonl")
    try
        cfg = solver_config(problem, accuracy, scf)
        fitted_projection = use_subspace && o.projection == :qtt_gradient
        v = Q.timed(rec, "potential_tci") do
            Q.cross(
                rec,
                grid.R,
                j -> potential[j + 1];
                tol = fitted_projection ? o.matrix_tolerance : accuracy.external_tolerance,
                cap = field_caps(accuracy).effective_potential_cap,
                label = "fixed_potential",
            )
        end
        compression = Q.timed(rec, "potential_diagnostics") do
            ev = Q.evaluate(v, rec)
            err2 = 0.0
            maxerr = 0.0
            for j = 0:(ngrid - 1)
                err = abs(ev(j) - potential[j + 1])
                err2 += err^2
                maxerr = max(maxerr, err)
            end
            (
                max_absolute_error = maxerr,
                rms_absolute_error = sqrt(err2 / ngrid),
                relative_l2 = iszero(err2) ? 0.0 : sqrt(err2 / sum(abs2, potential)),
                max_bond = Q.rank(v),
                bytes = Q.storage(v),
            )
        end
        Q.timed(rec, "potential_output") do
            if dir !== nothing
                write_json(joinpath(dir, "potential-compression.json"), compression)
                serialize(joinpath(dir, "potential-cores.jls"), v.c)
            end
        end
        H, green = Q.timed(rec, "operator_construction") do
            H = Q.Hamiltonian(
                v,
                Q.diagonal(v),
                Q.shift_mpo(grid.R, 1),
                Q.shift_mpo(grid.R, -1),
                dx * sqrt(problem.mass),
            )
            H, o.method == :lanczos ? nothing : Q.green_mpo(grid.R, H.h, scf.green_shift)
        end
        project_matrices = if o.projection == :streaming_grid
            basis -> spectrum_grid_matrices(basis, H, potential, cfg, rec, o)
        else
            basis -> (Q.h_gram(basis, H; cfg), Q.gram(basis; cfg))
        end
        search_field = if use_subspace
            Q.synchronized(
                o.projection == :streaming_grid ? Q.sampled(j -> potential[j + 1], cfg.R) :
                Q.sampled(H.potential, rec),
            )
        else
            nothing
        end
        orbitals = if resumed === nothing
            Q.timed(rec, "initial_orbitals") do
                envelope = Q.cross(
                    rec,
                    grid.R,
                    j -> exp(
                        o.virtual_envelope_amplitude *
                        cos(2pi * j * dx / o.virtual_envelope_period),
                    );
                    tol = o.ritz_tolerance,
                    cap = accuracy.orbital_cap,
                    label = "virtual_envelope",
                )
                vcat(
                    initial.orbitals,
                    [Q.normalize(Q.phase_tt(envelope, k)) for k in o.virtual_wavevectors],
                )
            end
        else
            resumed
        end
        initial_tol =
            o.adaptive_ritz ? max(o.ritz_tolerance, o.max_ritz_tolerance) : o.ritz_tolerance
        initial_cfg = merge(cfg, (orthogonalization_tolerance = initial_tol,))
        subspace = nothing
        if use_subspace && o.method != :lanczos
            orbitals, eps, gram_defect = Q.subspace_ritz(
                orbitals,
                o.states,
                project_matrices,
                cfg,
                rec;
                tol = o.ritz_tolerance,
            )
            subspace = (; basis = orbitals, energies = eps, gram_defect)
        elseif o.method == :lanczos
            orbitals, gram_defect = Q.orthogonalize(orbitals, cfg, rec)
            eps = zeros(o.states)
        elseif resumed === nothing
            orbitals = Q.timed(rec, "initial_orthogonalization") do
                first(Q.orthogonalize(orbitals, initial_cfg, rec; force = true))
            end
            orbitals, eps = spectrum_ritz(orbitals, H, cfg, rec, o, potential; tol = initial_tol)
        else
            eps = zeros(o.states)
        end
        history = Any[]
        previous = nothing
        converged = false
        for iteration = 0:(o.max_iterations)
            step_start = time_ns()
            metrics = nothing
            if iteration > 0
                rec.scf = iteration
                rec.inner = 1
                if o.method == :block_green
                    orbitals, metrics = simple_green_step(
                        orbitals,
                        last(history).eigenvalues,
                        H,
                        green,
                        cfg,
                        rec,
                        o,
                        potential,
                        accuracy,
                    )
                elseif o.method == :lrrap
                    orbitals, previous, metrics = Q.lrrap_step(
                        orbitals,
                        previous,
                        H,
                        green,
                        cfg,
                        rec;
                        work_tol = accuracy.work_tolerance,
                    )
                    # Rotation changes state identities; reset rather than mismatch columns.
                    orbitals, eps = spectrum_ritz(orbitals, H, cfg, rec, o, potential)
                    previous = nothing
                elseif use_subspace
                    orbitals, subspace, metrics = Q.subspace_step(
                        o.method,
                        orbitals,
                        subspace,
                        H,
                        green,
                        cfg,
                        rec;
                        work_tol = min(
                            accuracy.work_tolerance,
                            o.ritz_tolerance,
                            scf.eigen_tolerance / 10,
                        ),
                        ritz_tol = o.ritz_tolerance,
                        potential = search_field,
                        project_matrices,
                    )
                else
                    old_xs = orbitals
                    work =
                        o.adaptive_work ?
                        max(
                            accuracy.work_tolerance,
                            min(
                                o.max_work_tolerance,
                                o.work_residual_factor * last(history).max_residual,
                            ),
                        ) : accuracy.work_tolerance
                    ritz =
                        o.adaptive_ritz ?
                        max(
                            o.ritz_tolerance,
                            min(
                                o.max_ritz_tolerance,
                                o.ritz_residual_factor * H.h^2 * last(history).max_residual,
                            ),
                        ) : o.ritz_tolerance
                    orbitals, metrics = residual_subspace_step(
                        orbitals,
                        H,
                        green,
                        cfg,
                        rec,
                        o,
                        potential;
                        work_tolerance = work,
                        ritz_tolerance = ritz,
                        previous,
                        energies = last(history).eigenvalues,
                        diagnostic_residuals = last(history).residuals,
                    )
                    previous = old_xs
                end
            end
            rec.stage = "diagnostics"
            eps, residuals = Q.timed(rec, "residual_diagnostics") do
                spectrum_residuals(orbitals, H, cfg, o, rec, potential)
            end
            gram_all, gram_occ = Q.timed(rec, "overlap_diagnostics") do
                norm(Q.gram(orbitals; cfg) - I), norm(Q.gram(orbitals[1:(o.occupied_states)]; cfg) - I)
            end
            occupied_band_energy, energy_change = Q.timed(rec, "energy_diagnostics") do
                e = dot(problem.occupations, eps[1:(o.occupied_states)])
                e, isempty(history) ? nothing : abs(e - last(history).occupied_band_energy)
            end
            row = (
                iteration,
                seconds = (time_ns() - step_start) / 1e9,
                elapsed_seconds = (time_ns() - t0) / 1e9,
                eigenvalues = eps,
                residuals,
                max_residual = maximum(residuals),
                rms_residual = sqrt(sum(abs2, residuals) / o.states),
                occupied_max_residual = maximum(residuals[1:(o.occupied_states)]),
                occupied_rms_residual = sqrt(
                    sum(abs2, residuals[1:(o.occupied_states)]) / o.occupied_states,
                ),
                gram_defect = gram_all,
                occupied_gram_defect = gram_occ,
                trace = sum(eps),
                occupied_trace = sum(eps[1:(o.occupied_states)]),
                occupied_band_energy,
                energy_change,
                max_bond = maximum(Q.rank.(orbitals)),
                orbital_bytes = sum(Q.storage, orbitals),
                update = metrics,
            )
            push!(history, row)
            Q.event!(io, row)
            Q.timed(rec, "checkpoint_output") do
                spectrum_checkpoint(dir, problem, orbitals, eps, history, o, potential_hash)
            end
            @printf(
                "FIXED-H %3d states=%d occupied=%d maxres=%.3e occmax=%.3e rms=%.3e Gram=%.3e trace=%.10f rank=%d time=%.2fs\n",
                iteration,
                o.states,
                o.occupied_states,
                row.max_residual,
                row.occupied_max_residual,
                row.rms_residual,
                gram_all,
                row.trace,
                row.max_bond,
                row.seconds
            )
            flush(stdout)
            energy_change === nothing || (
                @printf(
                    "    occupied fixed-H band energy=%.12f Ha delta=%.3e Ha criterion=%s\n",
                    occupied_band_energy,
                    energy_change,
                    criterion,
                );
                flush(stdout)
            )
            accepted = if use_subspace
                iteration >= o.min_iterations &&
                    row.max_residual < scf.eigen_tolerance &&
                    row.gram_defect < scf.gram_tolerance
            else
                spectrum_accepted(o, iteration, energy_change)
            end
            if accepted
                converged = true
                break
            end
            dir !== nothing && isfile(joinpath(dir, "STOP")) && break
        end
        elapsed = (time_ns() - t0) / 1e9
        dir === nothing || write_json(
            joinpath(dir, "status.json"),
            (
                status = "finished",
                converged,
                criterion,
                iterations = last(history).iteration,
                elapsed_seconds = elapsed,
                potential_compression = compression,
                final = last(history),
                fixed_hamiltonian = true,
                solved_states = o.states,
                occupied_states = o.occupied_states,
                electrons = sum(problem.occupations),
                canonical_potential = use_subspace && o.projection == :qtt_gradient ?
                                      "fitted potential in subspace updates and Ritz; original input in diagnostics" :
                                      o.method == :lrrap ?
                                      "compressed potential in LRRAP update; original input in Ritz and diagnostics" :
                                      "original input samples in updates, Rayleigh-Ritz and residual diagnostics",
                method = string(o.method),
            ),
        )
        FixedSpectrumResult(
            problem,
            orbitals,
            eps,
            OrbitalState(problem, orbitals[1:(o.occupied_states)]),
            history,
            converged,
            elapsed,
        )
    catch e
        fail_run(dir, e)
        rethrow()
    finally
        close_run_io(io)
        close(rec)
    end
end

fixed_spectrum(::OrbitalState{<:Problem{<:AbstractModel3D}}, ::AbstractVector{<:Real}; kwargs...) =
    throw(ArgumentError("The fixed-spectrum interface is currently 1D only"))
