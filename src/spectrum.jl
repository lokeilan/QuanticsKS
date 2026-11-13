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
    direction_norm_floor::Float64 = 1e-12
end
struct FixedSpectrumResult
    problem::Problem
    orbitals::Vector{Q.QTT}
    eigenvalues::Vector{Float64}
    occupied::OrbitalState
    history::Vector{Any}
    accepted::Bool
    elapsed_seconds::Float64
end
function validate(spectrum::FixedSpectrumOptions)
    0 < spectrum.occupied_states <= spectrum.states && spectrum.max_iterations > 0 ||
        throw(ArgumentError("Invalid spectrum dimensions"))
    all(
        x -> isfinite(x) && x > 0,
        (
            spectrum.ritz_tolerance,
            spectrum.matrix_tolerance,
            spectrum.diagnostic_tolerance,
            spectrum.virtual_envelope_period,
        ),
    ) || throw(ArgumentError("Invalid fixed spectrum controls"))
    isfinite(spectrum.virtual_envelope_amplitude) ||
        throw(ArgumentError("Invalid virtual envelope"))
    spectrum.method in (:lrrap, :block_residual, :block_green, SUBSPACE_SOLVERS...) ||
        throw(ArgumentError("Unknown fixed spectrum method"))
    spectrum.search_basis in (:projected, :direct) || throw(ArgumentError("Unknown search basis"))
    spectrum.projection in (:streaming_grid, :qtt_gradient) &&
        spectrum.diagnostics in (:streaming_grid, :qtt) &&
        spectrum.grid_block_size > 0 ||
        throw(ArgumentError("Invalid projection/diagnostic controls"))
    all(
        x -> isfinite(x) && x > 0,
        (
            spectrum.direction_relative_cutoff,
            spectrum.direction_absolute_cutoff,
            spectrum.direction_residual_floor,
            spectrum.previous_projection_floor,
            spectrum.max_work_tolerance,
            spectrum.work_residual_factor,
            spectrum.max_ritz_tolerance,
            spectrum.ritz_residual_factor,
        ),
    ) || throw(ArgumentError("Invalid residual-subspace controls"))
    all(
        x -> isfinite(x) && x > 0,
        (
            spectrum.metric_relative_cutoff,
            spectrum.metric_absolute_cutoff,
            spectrum.green_tolerance_factor,
            spectrum.direction_norm_floor,
        ),
    ) || throw(ArgumentError("Invalid metric or Green rounding controls"))
    length(spectrum.virtual_wavevectors) == spectrum.states - spectrum.occupied_states ||
        throw(ArgumentError("One analytic wavevector is required for each added virtual state"))
    spectrum
end

# Fixed-H Ritz matrices and residuals using QTT or chunked grid integration.

"Exact finite-grid quadrature in bounded blocks; never stores an N-by-state orbital array."
function spectrum_grid_matrices(
    orbitals,
    H,
    potential,
    cfg,
    rec,
    spectrum;
    coefficients = nothing,
    overlap_only = false,
)
    ngrid = 2^cfg.R
    b = coefficients === nothing ? length(orbitals) : size(coefficients, 2)
    es = [Q.evaluate(x, rec) for x in orbitals]
    A = overlap_only ? nothing : zeros(ComplexF64, b, b)
    S = zeros(ComplexF64, b, b)
    for start = 0:(spectrum.grid_block_size):(ngrid - 1)
        count = min(spectrum.grid_block_size, ngrid - start)
        last_sample = overlap_only ? count - 1 : count
        B = Q.independent_columns(last_sample + 1, length(orbitals), cfg) do i
            [es[i](mod(start + j, ngrid)) for j = 0:last_sample]
        end
        coefficients === nothing || (B = B * coefficients)
        X = view(B, 1:count, :)
        S .+= X' * X
        if !overlap_only
            D = view(B, 2:(count + 1), :) - X
            A .+= X' * (view(potential, (start + 1):(start + count)) .* X) + (D' * D) / (2H.h^2)
        end
    end
    A, S
end
function spectrum_grid_residuals(orbitals, H, potential, cfg, rec, spectrum)
    ngrid = 2^cfg.R
    b = length(orbitals)
    es = [Q.evaluate(x, rec) for x in orbitals]
    energies = zeros(b)
    norms = zeros(b)
    for start = 0:(spectrum.grid_block_size):(ngrid - 1)
        count = min(spectrum.grid_block_size, ngrid - start)
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
    for start = 0:(spectrum.grid_block_size):(ngrid - 1)
        count = min(spectrum.grid_block_size, ngrid - start)
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
    spectrum,
    potential;
    keep = length(orbitals),
    tol = spectrum.ritz_tolerance,
    remove_dependent_directions = false,
)
    if spectrum.projection == :streaming_grid
        A, S = Q.timed(rec, "ritz_projection") do
            spectrum_grid_matrices(orbitals, H, potential, cfg, rec, spectrum)
        end
    else
        A, S = Q.timed(rec, "ritz_projection") do
            Q.h_gram(orbitals, H; cfg), Q.gram(orbitals; cfg)
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
        schur = if spectrum.projection == :streaming_grid
            last(
                spectrum_grid_matrices(
                    orbitals,
                    H,
                    potential,
                    cfg,
                    rec,
                    spectrum;
                    coefficients = projector,
                    overlap_only = true,
                ),
            )
        else
            S[(keep + 1):end, (keep + 1):end] - XZ' * projection
        end
        W = Q.canonical_orthogonalizer(
            schur,
            spectrum.metric_relative_cutoff,
            max(spectrum.direction_norm_floor^2, spectrum.metric_absolute_cutoff),
        )
        C = zeros(ComplexF64, length(orbitals), keep + size(W, 2))
        C[1:keep, 1:keep] = I(keep)
        C[1:keep, (keep + 1):end] = -projection * W
        C[(keep + 1):end, (keep + 1):end] = W
        reduced, reduced_metric = if spectrum.projection == :streaming_grid
            # Form projected values before quadrature: C'AC loses digits when
            # history directions nearly coincide with the accepted subspace.
            spectrum_grid_matrices(orbitals, H, potential, cfg, rec, spectrum; coefficients = C)
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
function spectrum_residuals(orbitals, H, cfg, spectrum, rec, potential)
    spectrum.diagnostics == :streaming_grid &&
        return spectrum_grid_residuals(orbitals, H, potential, cfg, rec, spectrum)
    measured = Q.recorded_map(Tuple{Float64,Float64}, length(orbitals), cfg, rec) do i, localrec
        x = orbitals[i]
        norm_squared = real(Q.inner(x, x))
        e = Q.h_expectation(H, x) / norm_squared
        residual =
            Q.ttnorm(
                Q.h_residual(
                    H,
                    x,
                    e,
                    cfg,
                    localrec;
                    tol = spectrum.diagnostic_tolerance,
                    cap = cfg.work_cap,
                    label = "fixed_diagnostic_residual_$i",
                    potential = j -> potential[j + 1],
                ),
            ) / sqrt(norm_squared)
        e, residual
    end
    first.(measured), last.(measured)
end

# Green/residual subspace updates at fixed H, with diagnostics for the updated orbitals.

"One Green update per orbital followed by one Rayleigh–Ritz in the updated block."
function simple_green_step(orbitals, energies, H, green, cfg, rec, spectrum, potential, accuracy)
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
            amplitude_norm = Q.ttnorm(y)
            amplitude_norm > 1e-14 || error("Green update produced a zero orbital")
            Q.scale(y, inv(amplitude_norm))
        end
    end
    rec.stage = "rayleigh_ritz"
    updated_orbitals, _ = spectrum_ritz(
        updated_orbitals,
        H,
        cfg,
        rec,
        spectrum,
        potential;
        tol = spectrum.ritz_tolerance,
    )
    updated_orbitals,
    (
        trial_dimension = length(orbitals),
        rhs_tolerance = accuracy.green_rhs_tolerance,
        green_tolerance = accuracy.algebra_tolerance,
        ritz_tolerance = spectrum.ritz_tolerance,
        search_basis = "updated orbitals only",
    )
end

"Green-preconditioned residual expansion; all trial and accepted vectors remain QTT."
function residual_subspace_step(
    orbitals,
    H,
    green,
    cfg,
    rec,
    spectrum,
    potential;
    work_tolerance,
    ritz_tolerance = spectrum.ritz_tolerance,
    previous = nothing,
)
    rec.stage = "residual_actions"
    candidates = Q.recorded_map(Union{Nothing,Q.QTT}, length(orbitals), cfg, rec) do i, localrec
        x = orbitals[i]
        e = Q.h_expectation(H, x) / real(Q.inner(x, x))
        r = Q.h_residual(
            H,
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
        Q.ttnorm(r) > spectrum.direction_residual_floor || return nothing
        # Green removes high-frequency action error; its own truncation must not
        # reintroduce it at the much looser action interpolation tolerance.
        green_tol =
            spectrum.search_basis == :direct ?
            min(work_tolerance, spectrum.green_tolerance_factor * ritz_tolerance) :
            work_tolerance
        z = Q.apply_mpo(green, r; tol = green_tol, cap = cfg.work_cap)
        amplitude_norm = Q.ttnorm(z)
        amplitude_norm > 1e-14 ? Q.scale(z, inv(amplitude_norm)) : nothing
    end
    zs = Q.QTT[z for z in candidates if z !== nothing]
    residual_count = length(zs)
    spectrum.use_previous_subspace && previous !== nothing && append!(zs, previous)
    isempty(zs) && return orbitals, (directions = 0, trial_dimension = length(orbitals))
    if spectrum.search_basis == :direct
        # Equivalent subspace without fitting projected, orthonormal directions.
        # Resolve linear dependence in the small overlap matrix at Rayleigh–Ritz.
        trial = vcat(orbitals, zs)
        rec.stage = "rayleigh_ritz"
        updated_orbitals, _ = spectrum_ritz(
            trial,
            H,
            cfg,
            rec,
            spectrum,
            potential;
            keep = spectrum.states,
            tol = ritz_tolerance,
            remove_dependent_directions = true,
        )
        return updated_orbitals,
        (
            directions = residual_count,
            trial_dimension = length(trial),
            work_tolerance,
            ritz_tolerance,
            green_tolerance = min(work_tolerance, spectrum.green_tolerance_factor * ritz_tolerance),
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
            remaining < spectrum.previous_projection_floor && return nothing
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
        amplitude_norm = Q.ttnorm(q)
        amplitude_norm > 1e-12 ? Q.scale(q, inv(amplitude_norm)) : nothing
    end
    projected = Q.QTT[z for z in candidates if z !== nothing]
    isempty(projected) && return orbitals, (directions = 0, trial_dimension = length(orbitals))
    S = Q.gram(projected; cfg)
    f = eigen(Hermitian((S + S') / 2))
    good = findall(
        >(
            max(
                maximum(f.values) * spectrum.direction_relative_cutoff,
                spectrum.direction_absolute_cutoff,
            ),
        ),
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
    updated_orbitals, _ = spectrum_ritz(
        trial,
        H,
        cfg,
        rec,
        spectrum,
        potential;
        keep = spectrum.states,
        tol = ritz_tolerance,
    )
    updated_orbitals,
    (
        directions = length(directions),
        trial_dimension = length(trial),
        trial_gram_defect = norm(Q.gram(trial; cfg) - I),
        work_tolerance,
        ritz_tolerance,
    )
end

# Fixed-potential iteration and checkpoints; accept reconstructed orbital residuals and Gram.

function spectrum_checkpoint(dir, problem, orbitals, energies, history, spectrum)
    dir === nothing && return
    data = Dict(
        "schema_version" => 2,
        "kind" => "fixed_spectrum",
        "problem" => problem_descriptor(problem),
        "cores" => [x.c for x in orbitals],
        "eigenvalues" => energies,
        "occupied_states" => spectrum.occupied_states,
        "history" => history,
        "potential_file" => "potential.jls",
    )
    tmp = joinpath(dir, "spectrum.jls.tmp")
    serialize(tmp, data)
    mv(tmp, joinpath(dir, "spectrum.jls"); force = true)
    save_checkpoint(
        joinpath(dir, "occupied-checkpoint.jls"),
        OrbitalState(problem, orbitals[1:(spectrum.occupied_states)]);
        metadata = Dict(
            "fixed_hamiltonian" => true,
            "solved_states" => spectrum.states,
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
    if solver !== nothing
        solver isa Symbol || throw(ArgumentError("solver must be a method Symbol"))
        spectrum = FixedSpectrumOptions(; merge(option_tuple(spectrum), (; method = solver))...)
    end
    validate(spectrum)
    representation(initial) == :qtt || throw(ArgumentError("Fixed spectrum requires QTT orbitals"))
    validate(accuracy)
    validate(scf)
    validate(tci)
    use_subspace = spectrum.method in SUBSPACE_SOLVERS
    problem = initial.problem
    grid = problem.grid
    ngrid = length(grid)
    dx = spacing(grid)
    length(potential) == ngrid && all(isfinite, potential) ||
        throw(ArgumentError("Invalid fixed potential"))
    length(initial.orbitals) == length(problem.occupations) == spectrum.occupied_states ||
        throw(ArgumentError("Physical occupations must describe only the selected occupied states"))
    spectrum.states <= ngrid || throw(ArgumentError("More states than grid points"))
    resumed = nothing
    if resume_from !== nothing
        data = deserialize(resume_from)
        get(data, "kind", nothing) == "fixed_spectrum" && get(data, "schema_version", 0) == 2 ||
            throw(ArgumentError("Not a supported spectrum checkpoint"))
        data["problem"] == problem_descriptor(problem) &&
            data["occupied_states"] == spectrum.occupied_states &&
            length(data["cores"]) == spectrum.states ||
            throw(ArgumentError("Resume problem or state count mismatch"))
        saved_potential = deserialize(joinpath(dirname(resume_from), data["potential_file"]))
        saved_potential == potential || throw(ArgumentError("Resume fixed potential mismatch"))
        resumed = [Q.QTT(c) for c in data["cores"]]
    end
    t0 = time_ns()
    Random.seed!(spectrum.seed)
    dir = prepare_run(
        output,
        "fixed_spectrum",
        problem;
        spectrum = spectrum,
        accuracy = accuracy,
        solver = scf,
        tci,
    )
    rec = recorder(dir, tci; parallel_tasks = scf.parallel_tasks)
    io = run_io(dir, "iterations.jsonl")
    try
        cfg = solver_config(problem, accuracy, scf)
        fitted_projection = use_subspace && spectrum.projection == :qtt_gradient
        v = Q.timed(rec, "potential_tci") do
            Q.cross(
                rec,
                grid.R,
                j -> potential[j + 1];
                tol = fitted_projection ? spectrum.matrix_tolerance : accuracy.external_tolerance,
                cap = field_caps(accuracy).effective_potential_cap,
                label = "fixed_potential",
            )
        end
        compression =
            dir === nothing ? nothing :
            Q.timed(rec, "potential_diagnostics") do
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
                serialize(joinpath(dir, "potential.jls"), collect(potential))
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
            H,
            spectrum.method == :block_lanczos ? nothing : Q.green_mpo(grid.R, H.h, scf.green_shift)
        end
        # Reuse fixed fields across iterations and matching projection/diagnostic budgets.
        projection_H =
            (spectrum.projection == :streaming_grid && spectrum.method != :block_residual) ||
            fitted_projection ? H :
            spectrum_projection_hamiltonian(H, potential, cfg, rec; tol = spectrum.matrix_tolerance)
        diagnostic_H =
            spectrum.diagnostics == :streaming_grid ? H :
            (spectrum.projection == :qtt_gradient || spectrum.method == :block_residual) &&
            spectrum.diagnostic_tolerance == spectrum.matrix_tolerance ? projection_H :
            spectrum_projection_hamiltonian(
                H,
                potential,
                cfg,
                rec;
                tol = spectrum.diagnostic_tolerance,
            )
        project_matrices = if spectrum.projection == :streaming_grid
            basis -> spectrum_grid_matrices(basis, H, potential, cfg, rec, spectrum)
        else
            basis -> (Q.h_gram(basis, projection_H; cfg), Q.gram(basis; cfg))
        end
        search_field = if use_subspace
            Q.synchronized(
                spectrum.projection == :streaming_grid ?
                Q.sampled(j -> potential[j + 1], cfg.R) : Q.sampled(H.potential, rec),
            )
        else
            nothing
        end
        orbitals = if resumed === nothing
            Q.timed(rec, "initial_orbitals") do
                isempty(spectrum.virtual_wavevectors) && return copy(initial.orbitals)
                envelope = Q.cross(
                    rec,
                    grid.R,
                    j -> exp(
                        spectrum.virtual_envelope_amplitude *
                        cos(2pi * j * dx / spectrum.virtual_envelope_period),
                    );
                    tol = spectrum.ritz_tolerance,
                    cap = accuracy.orbital_cap,
                    label = "virtual_envelope",
                )
                vcat(
                    initial.orbitals,
                    [Q.normalize(Q.phase_tt(envelope, k)) for k in spectrum.virtual_wavevectors],
                )
            end
        else
            resumed
        end
        initial_tol =
            spectrum.adaptive_ritz ? max(spectrum.ritz_tolerance, spectrum.max_ritz_tolerance) :
            spectrum.ritz_tolerance
        initial_cfg = merge(cfg, (orthogonalization_tolerance = initial_tol,))
        subspace = nothing
        if use_subspace
            orbitals = first(Q.orthogonalize(orbitals, cfg, rec))
        elseif resumed === nothing
            orbitals = Q.timed(rec, "initial_orthogonalization") do
                first(Q.orthogonalize(orbitals, initial_cfg, rec; force = true))
            end
            orbitals, _ = spectrum_ritz(
                orbitals,
                projection_H,
                cfg,
                rec,
                spectrum,
                potential;
                tol = initial_tol,
            )
        end
        history = Any[]
        previous = nothing
        accepted = false
        local energies
        for iteration = 0:(spectrum.max_iterations)
            step_start = time_ns()
            metrics = nothing
            if iteration > 0
                rec.scf = iteration
                rec.inner = 1
                if spectrum.method == :block_green
                    orbitals, metrics = simple_green_step(
                        orbitals,
                        last(history).eigenvalues,
                        projection_H,
                        green,
                        cfg,
                        rec,
                        spectrum,
                        potential,
                        accuracy,
                    )
                elseif spectrum.method == :lrrap
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
                    orbitals, _ =
                        spectrum_ritz(orbitals, projection_H, cfg, rec, spectrum, potential)
                    previous = nothing
                elseif spectrum.method == :block_lanczos
                    orbitals, subspace, metrics = Q.block_lanczos_step(
                        orbitals,
                        subspace,
                        H,
                        cfg,
                        rec;
                        work_tol = min(
                            accuracy.work_tolerance,
                            spectrum.ritz_tolerance,
                            scf.eigen_tolerance / 10,
                        ),
                        ritz_tol = spectrum.ritz_tolerance,
                        potential = search_field,
                    )
                elseif use_subspace
                    orbitals, subspace, metrics = Q.block_davidson_step(
                        orbitals,
                        subspace,
                        H,
                        green,
                        cfg,
                        rec;
                        work_tol = min(
                            accuracy.work_tolerance,
                            spectrum.ritz_tolerance,
                            scf.eigen_tolerance / 10,
                        ),
                        ritz_tol = spectrum.ritz_tolerance,
                        potential = search_field,
                        project_matrices,
                    )
                else
                    previous_orbitals = orbitals
                    work =
                        spectrum.adaptive_work ?
                        max(
                            accuracy.work_tolerance,
                            min(
                                spectrum.max_work_tolerance,
                                spectrum.work_residual_factor * last(history).max_residual,
                            ),
                        ) : accuracy.work_tolerance
                    ritz =
                        spectrum.adaptive_ritz ?
                        max(
                            spectrum.ritz_tolerance,
                            min(
                                spectrum.max_ritz_tolerance,
                                spectrum.ritz_residual_factor * H.h^2 * last(history).max_residual,
                            ),
                        ) : spectrum.ritz_tolerance
                    orbitals, metrics = residual_subspace_step(
                        orbitals,
                        projection_H,
                        green,
                        cfg,
                        rec,
                        spectrum,
                        potential;
                        work_tolerance = work,
                        ritz_tolerance = ritz,
                        previous,
                    )
                    previous = previous_orbitals
                end
            end
            rec.stage = "diagnostics"
            energies, residuals = Q.timed(rec, "residual_diagnostics") do
                spectrum_residuals(orbitals, diagnostic_H, cfg, spectrum, rec, potential)
            end
            gram_all, gram_occ = Q.timed(rec, "overlap_diagnostics") do
                overlap = Q.gram(orbitals; cfg)
                occupied =
                    view(overlap, 1:(spectrum.occupied_states), 1:(spectrum.occupied_states))
                norm(overlap - I), norm(occupied - I)
            end
            occupied_band_energy, energy_change = Q.timed(rec, "energy_diagnostics") do
                e = dot(problem.occupations, energies[1:(spectrum.occupied_states)])
                e, isempty(history) ? nothing : abs(e - last(history).occupied_band_energy)
            end
            row = (
                iteration,
                seconds = (time_ns() - step_start) / 1e9,
                elapsed_seconds = (time_ns() - t0) / 1e9,
                eigenvalues = energies,
                residuals,
                max_residual = maximum(residuals),
                rms_residual = sqrt(sum(abs2, residuals) / spectrum.states),
                occupied_max_residual = maximum(residuals[1:(spectrum.occupied_states)]),
                occupied_rms_residual = sqrt(
                    sum(abs2, residuals[1:(spectrum.occupied_states)]) / spectrum.occupied_states,
                ),
                gram_defect = gram_all,
                occupied_gram_defect = gram_occ,
                trace = sum(energies),
                occupied_trace = sum(energies[1:(spectrum.occupied_states)]),
                occupied_band_energy,
                energy_change,
                max_bond = maximum(Q.rank.(orbitals)),
                orbital_bytes = sum(Q.storage, orbitals),
                update = metrics,
            )
            push!(history, row)
            Q.event!(io, row)
            Q.timed(rec, "checkpoint_output") do
                spectrum_checkpoint(dir, problem, orbitals, energies, history, spectrum)
            end
            @printf(
                "FIXED-H %3d states=%d occupied=%d maxres=%.3e occmax=%.3e rms=%.3e Gram=%.3e trace=%.10f rank=%d time=%.2fs\n",
                iteration,
                spectrum.states,
                spectrum.occupied_states,
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
                    "    occupied fixed-H band energy=%.12f Ha delta=%.3e Ha \n",
                    occupied_band_energy,
                    energy_change,
                );
                flush(stdout)
            )
            accepted =
                row.max_residual < scf.eigen_tolerance && row.gram_defect < scf.gram_tolerance
            accepted && break
            dir !== nothing && isfile(joinpath(dir, "STOP")) && break
        end
        elapsed = (time_ns() - t0) / 1e9
        dir === nothing || write_json(
            joinpath(dir, "status.json"),
            (
                status = "finished",
                accepted,
                iterations = last(history).iteration,
                elapsed_seconds = elapsed,
                potential_compression = compression,
                final = last(history),
                fixed_hamiltonian = true,
                solved_states = spectrum.states,
                occupied_states = spectrum.occupied_states,
                electrons = sum(problem.occupations),
                projection = string(spectrum.projection),
                diagnostics = string(spectrum.diagnostics),
                potential_file = "potential.jls",
                method = string(spectrum.method),
            ),
        )
        FixedSpectrumResult(
            problem,
            orbitals,
            energies,
            OrbitalState(problem, orbitals[1:(spectrum.occupied_states)]),
            history,
            accepted,
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
