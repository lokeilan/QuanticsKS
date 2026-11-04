module HamiltonianFieldChecks
using Test, QuanticsKS, LinearAlgebra, Random, JSON3
const K = QuanticsKS
const Q = K.Q
const measurements = Dict{String,Any}()
BLAS.set_num_threads(1)

values(x) = [
    only(prod(g[:, ((j >> (k - 1)) & 1) + 1, :] for (k, g) in enumerate(x.c))) for
    j = 0:(2^length(x) - 1)
]
function dense_mpo(w)
    n = 1 << length(w)
    [
        only(
            prod(
                g[:, ((i >> (k - 1)) & 1) + 1, ((j >> (k - 1)) & 1) + 1, :] for
                (k, g) in enumerate(w)
            ),
        ) for i = 0:(n - 1), j = 0:(n - 1)
    ]
end
function kinetic(R, h)
    n = 1 << R
    S = circshift(Matrix{Float64}(I, n, n), (1, 0))
    (2I - S - S') / (2h^2)
end

@testset "Carry shifts and discrete periodic Green inverse" begin
    shift_error = 0.0
    for R in (1, 2, 4), boundary in (:periodic, :open), step in (-1, 1)
        n = 1 << R
        expected = zeros(n, n)
        for j = 0:(n - 1)
            i = j + step
            boundary == :open && !(0 <= i < n) && continue
            expected[mod(i, n) + 1, j + 1] = 1
        end
        w = Q.shift_mpo(R, step; boundary)
        err = norm(dense_mpo(w) - expected)
        shift_error = max(shift_error, err)
        @test err == 0
        @test maximum(g -> max(size(g, 1), size(g, 4)), w) <= 2
    end
    inverse_errors = Float64[]
    for R in (2, 5), mass in (1.0, 1.7), mu in (0.2, 0.5, -20im)
        h = 8 / 2^R * sqrt(mass)
        G = dense_mpo(Q.green_mpo(R, h, mu))
        err = norm((kinetic(R, h) + mu * I) * G - I)
        push!(inverse_errors, err)
        @test err < 1e-10
    end
    @test_throws ArgumentError Q.shift_mpo(3, 2)
    @test_throws ArgumentError Q.green_mpo(3, 1.0, 0.0)
    measurements["operators"] = (;
        shift_max_absolute_error = shift_error,
        green_inverse_max_frobenius_defect = maximum(inverse_errors),
    )
end

@testset "Frozen density, fused local sources, and core field projections" begin
    Random.seed!(20261103)
    R = 6
    p = hydrogen_chain(R = R, natoms = 2, distance = 4.0)
    h, N = K.spacing(p.grid), length(p.grid)
    a = Accuracy(
        field_tolerance = 1e-11,
        density_tolerance = 1e-12,
        external_tolerance = 1e-13,
        kernel_tolerance = 1e-13,
        algebra_tolerance = 1e-13,
        xc_energy_tolerance = 1e-12,
        orbital_cap = 32,
        work_cap = 32,
        field_cap = 32,
    )
    cfg = K.solver_config(p, a)
    rec = K.recorder(nothing, TCIOptions())
    ops = K.build_operators(p, a, rec, PhysicsNumerics())
    n = Q.cross(rec, R, j -> 0.25 * (1 + 0.2cos(2pi * j / N)); tol = 1e-13, cap = 32)
    nv = values(n)
    calls = rec.calls
    local_only = K.local_potential_samples(p, n, ops, a, rec)
    @test rec.calls == calls
    @test isfinite(local_only(0))
    calls = rec.calls
    H = K.effective_hamiltonian(p, n, ops, a, rec)
    @test rec.calls == calls + 1
    density_sample = H.samples.input.arguments[3]
    @test density_sample.input.evaluator.x.c[1] !== n.c[1]
    # Changing the caller's density cannot change the fixed Hamiltonian.
    n.c[1] .*= 1.5
    kernel = [K.kernel_value(p.model, mod(i - j, N) * h, p.grid.L) for i = 0:(N - 1), j = 0:(N - 1)]
    vh = h * kernel * nv
    external = [K.external_value(p.model, j * h, p.grid.L) for j = 0:(N - 1)]
    expected = external + vh + [Q.xc(ops.xc_table, real(v))[2] for v in nv]
    field = Q.potential_samples(H, rec)
    sampled = [field(j) for j = 0:(N - 1)]
    field_error = maximum(abs, sampled - expected)
    representation_error = maximum(abs, values(H.potential) - expected)
    @test field_error < 2e-10
    @test representation_error < 2e-9
    A = kinetic(R, h) + Diagonal(expected)
    base = Q.normalize(Q.constant_tt(R))
    xs = [
        Q.normalize(Q.add(base, Q.phase_tt(base, 2); b = 0.2im, tol = 0.0, cap = 8)),
        Q.phase_tt(base, 1),
    ]
    X = hcat(values.(xs)...)
    energies = [real(dot(x, A * x)) for x in eachcol(X)]
    evaluations = field.evaluations
    residual_errors = Float64[]
    for (i, x) in enumerate(xs)
        r = Q.h_residual(H, x, energies[i], cfg, rec; tol = 1e-12, cap = 32)
        err = norm(values(r) - (A - energies[i] * I) * X[:, i])
        push!(residual_errors, err)
        @test err < 2e-8
    end
    green = Q.green_mpo(R, h, cfg.green_shift)
    G = inv(kinetic(R, h) + cfg.green_shift * I)
    ys = Q.block_green(xs, energies, H, green, cfg, rec)
    green_errors = Float64[]
    for i in eachindex(xs)
        target = G * ((cfg.green_shift + energies[i] .- expected) .* X[:, i])
        target ./= norm(target)
        err = norm(values(ys[i]) - target)
        push!(green_errors, err)
        @test err < 2e-8
    end
    @test field.evaluations == evaluations
    @test Q.potential_samples(H, rec) === field
    projected_error = norm(Q.h_gram(xs, H) - X' * A * X)
    @test projected_error < 2e-8
    nout = K.density(p, [xs[1]], a, rec; renormalize = true)
    rho = values(nout)
    energy = K.total_energy(p, [xs[1]], nout, ops, a, rec)
    xc_energy = h * sum(Q.xc(ops.xc_table, real(v))[1] for v in rho)
    @test abs(energy.xc - xc_energy) < 2e-9
    @test abs(energy.nuclear - K.nuclear_energy(p.model, p.grid.L)) < 1e-14
    measurements["local_field"] = (;
        pointwise_max_absolute_error = field_error,
        core_representation_max_absolute_error = representation_error,
        residual_max_l2_error = maximum(residual_errors),
        green_update_max_l2_error = maximum(green_errors),
        projected_hamiltonian_frobenius_error = projected_error,
        xc_energy_absolute_error_Ha = abs(energy.xc - xc_energy),
        shared_field_evaluations = evaluations,
        extra_evaluations_for_repeated_orbital_actions = field.evaluations - evaluations,
        fixed_density_owns_cores = true,
    )
    close(rec)
end

@testset "TCI post-fit rounding uses the configured factor" begin
    mktempdir() do dir
        rec = K.recorder(dir, TCIOptions())
        f(j) = exp(0.2cos(2pi * j / 64)) + 0.1im * sin(4pi * j / 64)
        x = Q.cross(rec, 6, f; tol = 1e-10, cap = 16, label = "construction")
        y = Q.cross(
            rec,
            6,
            f;
            tol = 1e-10,
            cap = 16,
            label = "explicit_rounding",
            rounding_tolerance = 1e-12,
        )
        raw = Q.cross(
            rec,
            6,
            f;
            tol = 1e-10,
            cap = 16,
            label = "unrounded_diagnostic",
            rounding_tolerance = nothing,
        )
        @test norm(values(x) - f.(0:63)) < 1e-8
        @test norm(values(y) - f.(0:63)) < 1e-8
        @test norm(values(raw) - f.(0:63)) < 1e-8
        for g in x.c[2:end]
            A = reshape(g, size(g, 1), :)
            @test norm(A * A' - I) < 2e-12
        end
        close(rec)
        rows = JSON3.read.(readlines(joinpath(dir, "tci_calls.jsonl")))
        @test rows[1].post_rounding_seconds >= 0.0
        @test rows[1].post_rounding_tolerance ≈ 1e-11
        @test rows[2].post_rounding_tolerance == 1e-12
        @test rows[3].post_rounding_tolerance === nothing
        @test rows[3].post_rounding_seconds == 0.0
        measurements["construction"] = (;
            default_post_rounding_seconds = rows[1].post_rounding_seconds,
            default_post_rounding_tolerance = rows[1].post_rounding_tolerance,
            actual_rank = Q.rank(x),
            rank_cap = 16,
            analytic_l2_error = norm(values(x) - f.(0:63)),
        )
    end
end
end
