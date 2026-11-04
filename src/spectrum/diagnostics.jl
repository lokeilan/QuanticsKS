# Fixed-H Ritz matrices and residuals using QTT or chunked grid integration.

"Exact finite-grid quadrature in bounded blocks; never stores an N-by-state orbital array."
function spectrum_grid_matrices(xs, H, potential, cfg, rec, o; coefficients = nothing)
    N = 2^cfg.R
    b = coefficients === nothing ? length(xs) : size(coefficients, 2)
    es = [Q.evaluate(x, rec) for x in xs]
    A = zeros(ComplexF64, b, b)
    S = zeros(ComplexF64, b, b)
    for start = 0:(o.grid_block_size):(N - 1)
        count = min(o.grid_block_size, N - start)
        B = Q.independent_columns(count + 1, length(xs), cfg) do i
            [es[i](mod(start + j, N)) for j = 0:count]
        end
        coefficients === nothing || (B = B * coefficients)
        X = view(B, 1:count, :)
        D = view(B, 2:(count + 1), :) - X
        S .+= X' * X
        A .+= X' * (view(potential, (start + 1):(start + count)) .* X) + (D' * D) / (2H.h^2)
    end
    A, S
end
function spectrum_grid_residuals(xs, H, potential, cfg, rec, o)
    N = 2^cfg.R
    b = length(xs)
    es = [Q.evaluate(x, rec) for x in xs]
    energies = zeros(b)
    norms = zeros(b)
    for start = 0:(o.grid_block_size):(N - 1)
        count = min(o.grid_block_size, N - start)
        B = Q.independent_columns(count + 1, b, cfg) do i
            [es[i](mod(start + j, N)) for j = 0:count]
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
    for start = 0:(o.grid_block_size):(N - 1)
        count = min(o.grid_block_size, N - start)
        B = Q.independent_columns(count + 2, b, cfg) do i
            [es[i](mod(start + j, N)) for j = -1:count]
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
            cap = cfg.opcap,
            label = "fixed_projection_potential",
            pivots = [argmax(j -> abs(potential[j]), eachindex(potential)) - 1],
        )
    Q.Hamiltonian(v, Q.diagonal(v), H.plus, H.minus, H.h)
end

"Project directly from existing cores; reconstruct only the final Ritz orbitals."
function spectrum_ritz(
    xs,
    H,
    cfg,
    rec,
    o,
    potential;
    keep = length(xs),
    tol = o.ritz_tolerance,
    reveal_metric = false,
)
    if o.projection == :streaming_grid
        A, S = Q.timed(rec, "ritz_projection") do
            spectrum_grid_matrices(xs, H, potential, cfg, rec, o)
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
            Q.h_gram(xs, projected_H; cfg), Q.gram(xs; cfg)
        end
    end
    if reveal_metric
        # Preserve every current Ritz vector. Dropping eigenvectors of the raw
        # trial Gram can rotate away part of X and destroy monotonic convergence.
        XZ = view(S, 1:keep, (keep + 1):size(S, 2))
        projection = Hermitian(S[1:keep, 1:keep]) \ XZ
        projector = vcat(-projection, Matrix{ComplexF64}(I, length(xs) - keep, length(xs) - keep))
        schur = if o.projection == :streaming_grid
            last(spectrum_grid_matrices(xs, H, potential, cfg, rec, o; coefficients = projector))
        else
            S[(keep + 1):end, (keep + 1):end] - XZ' * projection
        end
        metric = eigen(Hermitian((schur + schur') / 2))
        good = findall(
            >(
                max(
                    o.direction_norm_floor^2,
                    o.metric_absolute_cutoff,
                    o.metric_relative_cutoff * maximum(metric.values),
                ),
            ),
            metric.values,
        )
        W = metric.vectors[:, good] * Diagonal(1 ./ sqrt.(metric.values[good]))
        C = zeros(ComplexF64, length(xs), keep + length(good))
        C[1:keep, 1:keep] = I(keep)
        C[1:keep, (keep + 1):end] = -projection * W
        C[(keep + 1):end, (keep + 1):end] = W
        reduced, reduced_metric = if o.projection == :streaming_grid
            # Form projected values before quadrature: C'AC loses digits when
            # history directions nearly coincide with the accepted subspace.
            spectrum_grid_matrices(xs, H, potential, cfg, rec, o; coefficients = C)
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
    ys = Q.timed(rec, "ritz_reconstruction") do
        oracle = Q.synchronized(Q.block_oracle(xs, rec))
        Q.recorded_map(Q.QTT, keep, cfg, rec) do i, localrec
            Q.linear_combination(
                localrec,
                xs,
                view(coefficients, :, i);
                oracle,
                tol,
                cap = cfg.rankcap,
                label = "fixed_ritz_$i",
            )
        end
    end
    ys, fac.values[1:keep]
end
function spectrum_actions(xs, H, cfg, rec, o, potential; tol = o.diagnostic_tolerance)
    # Compatibility entry for consumers explicitly requesting final H*x cores.
    # Search and diagnostic residuals bypass this intermediate representation.
    projected_H = spectrum_projection_hamiltonian(H, potential, cfg, rec; tol)
    Q.recorded_map(Q.QTT, length(xs), cfg, rec) do i, localrec
        Q.h_residual(
            projected_H,
            xs[i],
            0.0,
            cfg,
            localrec;
            tol,
            cap = cfg.workcap,
            label = "fixed_Hpsi_$i",
            potential = j -> potential[j + 1],
        )
    end
end
function spectrum_residuals(xs, H, cfg, o, rec, potential)
    o.diagnostics == :streaming_grid &&
        return spectrum_grid_residuals(xs, H, potential, cfg, rec, o)
    projected_H =
        spectrum_projection_hamiltonian(H, potential, cfg, rec; tol = o.diagnostic_tolerance)
    measured = Q.recorded_map(Tuple{Float64,Float64}, length(xs), cfg, rec) do i, localrec
        x = xs[i]
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
                    cap = cfg.workcap,
                    label = "fixed_diagnostic_residual_$i",
                    potential = j -> potential[j + 1],
                ),
            ) / Q.ttnorm(x)
        e, residual
    end
    first.(measured), last.(measured)
end
