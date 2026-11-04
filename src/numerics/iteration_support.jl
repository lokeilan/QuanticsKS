# Density mixing and output compression use separate working and storage tolerances.

function mix_density(oldinput, nin, inputs, residuals, cfg, rec)
    oldinput === nothing && return nin, "initial"
    residual = linear_expression((nin, oldinput), (1, -1))
    push!(inputs, oldinput)
    push!(residuals, residual)
    length(inputs) > cfg.anderson_history && (popfirst!(inputs); popfirst!(residuals))
    m = length(inputs)
    coeff = zeros(m)
    coeff[end] = 1
    method = "linear"
    if cfg.anderson && m >= 3
        B = timed(rec, "mixing_residual_gram") do
            real.(expression_gram(residuals; rec))
        end
        B .+= Diagonal(fill(maximum(diag(B)) * cfg.anderson_regularization + 1e-20, m))
        trial = B \ ones(m)
        trial ./= sum(trial)
        if all(isfinite, trial) && maximum(abs, trial) < cfg.anderson_coefficient_limit
            coeff = trial
            method = "anderson"
        end
    end
    active = findall(c -> abs(c) >= 1e-15, coeff)
    terms = vcat(inputs[active], residuals[active])
    weights = vcat(coeff[active], cfg.mixing .* coeff[active])
    # Flatten n_i + alpha*(rho_i-n_i); history holds only original QTT
    # leaves, so expression depth stays one and at most 2*history terms live.
    linear_expression(terms, weights), method
end

materialize_density(n::QTT, cfg, rec) = n
function materialize_density(n::QTTExpression, cfg, rec)
    timed(rec, "density_mixing") do
        mixed =
            materialize_expression(n; tol = get(cfg, :mixing_tol, cfg.round_tol), cap = cfg.opcap)
        h = cfg.cell / 2^cfg.R
        charge = h * real(inner(constant_tt(cfg.R), mixed; cfg))
        isfinite(charge) && charge > 0 || error("Invalid charge in mixed density: $charge")
        scale(mixed, cfg.electrons / charge)
    end
end

"Compress stored orbitals at their requested accuracy; internal operator algebra keeps its tighter tolerance."
function pack_orbitals(xs, cfg; tol = cfg.tci_tol, input_gram_defect = nothing)
    isfinite(tol) && tol >= 0 || throw(ArgumentError("orbital_tol must be finite and nonnegative"))
    before = sum(storage, xs)
    input_defect = input_gram_defect === nothing ? norm(gram(xs; cfg) - I) : input_gram_defect
    tol == 0 && return xs,
    (
        orbital_storage_tolerance = 0.0,
        orbital_bytes_before_pack = before,
        orbital_pack_attempts = 0,
        packed_gram_defect = input_defect,
    )
    for attempt = 1:(cfg.pack_attempts)
        actual = tol / 10^(attempt - 1)
        ys = Vector{QTT}(undef, length(xs))
        foreach_independent(length(xs), cfg) do i
            ys[i] = normalize(rounded(xs[i]; tol = actual, cap = cfg.rankcap))
        end
        # Compression must not sacrifice occupied-space orthogonality.
        defect = norm(gram(ys; cfg) - I)
        if defect <= max(cfg.pack_gram_tolerance, 2input_defect)
            return ys,
            (
                orbital_storage_tolerance = actual,
                orbital_bytes_before_pack = before,
                orbital_pack_attempts = attempt,
                packed_gram_defect = defect,
            )
        end
    end
    error("Orbital compression failed the Gram check at requested tolerances")
end
