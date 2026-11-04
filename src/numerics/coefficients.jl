# Optimize small coefficient matrices while enforcing orthogonality between orbitals.

"Alternating minimization with diagonal coefficients on old orbitals (paper Eq.23-29)."
function coefficients(
    Sxx,
    Vx,
    Vhx,
    HV,
    energies;
    nsweeps = 2,
    parallel_tasks = 1,
    metric_relative_cutoff = 1e-11,
    metric_absolute_cutoff = 1e-13,
    constraint_relative_cutoff = 1e-10,
    constraint_absolute_cutoff = 1e-12,
)
    all(isfinite, Sxx) &&
        all(isfinite, Vx) &&
        all(isfinite, Vhx) &&
        all(isfinite, HV) &&
        all(isfinite, energies) || error("Nonfinite input to coefficient minimizer")
    m, b = size(Vx)
    G = [Sxx Vx'; Vx Matrix{ComplexF64}(I, m, m)]
    coef = vcat(Matrix{ComplexF64}(I, b, b), zeros(ComplexF64, m, b))
    whites = Vector{Matrix{ComplexF64}}(undef, b)
    as = Vector{Matrix{ComplexF64}}(undef, b)
    foreach_independent(b, (; parallel_tasks)) do i
        local ids, a
        ids = vcat(i, (b + 1):(b + m))
        g = Hermitian(G[ids, ids])
        e = eigen(g)
        good = findall(
            >(max(maximum(e.values) * metric_relative_cutoff, metric_absolute_cutoff)),
            e.values,
        )
        whites[i] = e.vectors[:, good] * Diagonal(1 ./ sqrt.(e.values[good]))
        a = zeros(ComplexF64, m + 1, m + 1)
        a[1, 1] = energies[i] * real(Sxx[i, i])
        a[2:end, 1] = Vhx[:, i]
        a[1, 2:end] = conj.(Vhx[:, i])
        a[2:end, 2:end] = HV
        as[i] = a
    end
    for sweep = 1:nsweeps, i = 1:b
        ids = vcat(i, (b + 1):(b + m))
        other = [j for j = 1:b if j != i]
        w = whites[i]
        constraints = w' * (G[ids, :] * coef[:, other])
        s = stable_svd(constraints; full = true)
        rr = count(
            >(
                max(
                    isempty(s.S) ? 0.0 : maximum(s.S) * constraint_relative_cutoff,
                    constraint_absolute_cutoff,
                ),
            ),
            s.S,
        )
        q = s.U[:, (rr + 1):end]
        size(q, 2) > 0 || error("No feasible coefficient direction")
        z = w * q
        a = Hermitian(z' * as[i] * z)
        all(isfinite, a) || error(
            "Nonfinite reduced coefficient matrix at sweep=$sweep orbital=$i; whiten=$(maximum(abs,w)) null=$(size(q))",
        )
        ev = eigen(a)
        v = z * ev.vectors[:, 1]
        overlap = dot(G[ids, i], v)
        abs(overlap) > 1e-12 && (v .*= conj(overlap) / abs(overlap))
        coef[:, i] .= 0
        coef[ids, i] = v
    end
    trace_after = sum(
        real(dot(coef[vcat(i, (b + 1):(b + m)), i], as[i] * coef[vcat(i, (b + 1):(b + m)), i]))
        for i = 1:b
    )
    (
        c = diag(coef[1:b, :]),
        C = coef[(b + 1):end, :],
        gram_defect = norm(coef' * G * coef - I),
        trace_after = trace_after,
    )
end
