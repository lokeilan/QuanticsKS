using Test, QuanticsKS, LinearAlgebra, Random, QuadGK, SpecialFunctions, TOML, JSON3
const K = QuanticsKS
const Q3 = K.Q
BLAS.set_num_threads(1)
Random.seed!(711)
values3(x) = [Q3.Evaluator(x)(j) for j = 0:(2^length(x) - 1)]

function dense_kinetic3(g, mass = 1.0)
    n, h = 1 << g.R, K.spacing(g)
    T = zeros(length(g), length(g))
    for j = 0:(length(g) - 1)
        q = grid_indices(g, j)
        for axis = 1:3
            T[j + 1, j + 1] += 1 / (mass * h[axis]^2)
            for sign in (-1, 1)
                neighbor = ntuple(a -> q[a] + (a == axis ? sign : 0), 3)
                all(v -> 0 <= v < n, neighbor) || continue
                T[j + 1, grid_index(g, neighbor) + 1] -= 1 / (2mass * h[axis]^2)
            end
        end
    end
    T
end

@testset "Hartree contraction retains nonisometric MPO suffixes" begin
    g = BoxGrid3D(3, 8.0)
    p = Problem(g, AllElectronLDA3D([(0.0, 0.0, 0.0)], [2.0]), [2.0])
    a = Accuracy(field_cap = 64, orbital_cap = 32, work_cap = 64)
    rec = K.recorder(nothing, TCIOptions())
    ops = K.build_operators(p, a, rec, PhysicsNumerics())
    rho = Q3.cross(
        rec,
        nsites(g),
        j -> exp(-sum(abs2, grid_position(g, j)) / 3);
        tol = 1e-12,
        cap = 16,
    )
    evrho = Q3.Evaluator(rho)
    vh = K.hartree(rho, p, ops, a, rec)
    evh = Q3.Evaluator(vh)
    for j in (0, 73, 256, 511)
        r = grid_position(g, j)
        reference =
            volume_element(g) * sum(
                K.coulomb_cell_average(r .- grid_position(g, k), K.spacing(g)) * evrho(k) for
                k = 0:(length(g) - 1)
            )
        @test abs(evh(j) - reference) < 1e-9
    end
end

@testset "3D geometry, singular integrals, and LDA derivative" begin
    g = BoxGrid3D(3, (6, 8, 10))
    @test nsites(g) == 9
    @test grid_index(g, (1, 0, 0)) == 1
    @test grid_index(g, (0, 1, 0)) == 2
    @test grid_index(g, (0, 0, 1)) == 4
    @test all(j -> grid_index(g, grid_indices(g, j)) == j, 0:(length(g) - 1))
    @test volume_element(g) * length(g) ≈ 480
    @test_throws ArgumentError BoxGrid3D(20, 8)
    @test_throws ArgumentError Problem(g, LocalPotential1D(0), [1.0])
    h = (0.7, 1.1, 1.3)
    for r in ((0.0, 0.0, 0.0), (0.2, 0.7, -0.3), (0.7, 1.1, 1.3), (8.0, -4.0, 6.0))
        # Independent Laplace/Gaussian integral of the rectangular voxel.
        integrand(t) =
            t == 0 ? 2 / sqrt(pi) :
            2 / sqrt(pi) * prod(
                sqrt(pi) / (2t * h[a]) * (erf(t * (r[a] + h[a] / 2)) - erf(t * (r[a] - h[a] / 2)))
                for a = 1:3
            )
        reference = first(quadgk(integrand, 0.0, Inf; rtol = 1e-11))
        @test K.coulomb_cell_average(r, h) ≈ reference rtol = 3e-8
        @test K.coulomb_cell_average(-1 .* r, h) ≈ reference rtol = 3e-8
    end
    @test K.lda_pz81(0.0) == (0.0, 0.0)
    for rho in (1e-8, 0.01, 0.1, 1.0, 100.0)
        delta = 1e-5rho
        derivative = (K.lda_pz81(rho + delta)[1] - K.lda_pz81(rho - delta)[1]) / (2delta)
        @test K.lda_pz81(rho)[2] ≈ derivative rtol = 1e-8
    end
    p = hydrogen_cluster(R = 2)
    @test p.model.positions[2][1] - p.model.positions[1][1] ≈ 2K.ANGSTROM_TO_BOHR
    @test sum(p.occupations) == 12
end

@testset "Open Cartesian MPOs, sampled action, and discrete Green inverse" begin
    g = BoxGrid3D(2, (5, 6, 7))
    p = Problem(
        g,
        LocalPotential3D((x, y, z) -> 0.2x - 0.07y^2 + 0.03z; id = "test3"),
        [1.0];
        mass = 1.7,
    )
    a = Accuracy(orbital_cap = 16, field_cap = 32, work_cap = 32)
    rec = K.recorder(nothing, TCIOptions())
    phys = PhysicsNumerics(green_relative_tolerance = 1e-9)
    ops = K.build_operators(p, a, rec, phys)
    psi = Q3.normalize(
        Q3.cross(
            rec,
            nsites(g),
            j -> begin
                r = grid_position(g, j)
                exp(-sum(abs2, r) / 5) * (1 + 0.1im * r[2])
            end;
            tol = 1e-12,
            cap = 16,
        ),
    )
    rho = K.density(p, [psi], a, rec)
    @test volume_element(g) * real(sum(Q3.tensortrain(rho))) ≈ 1 atol = 1e-11
    H = K.effective_hamiltonian(p, rho, ops, a, rec)
    T = dense_kinetic3(g, p.mass)
    V = [p.model.potential(grid_position(g, j)...) for j = 0:(length(g) - 1)]
    Hd = T + Diagonal(V)
    cfg = K.solver_config(p, a)
    densepsi = values3(psi)
    action = values3(Q3.h_action(H, psi, cfg, rec))
    @test norm(action - Hd * densepsi) < 1e-10
    @test Q3.h_expectation(H, psi) ≈ real(dot(densepsi, Hd * densepsi)) atol = 1e-11
    for axis = 1:3
        origin = Q3.delta_tt(nsites(g), grid_index(g, ntuple(a -> a == axis ? 3 : 1, 3)))
        @test Q3.ttnorm(Q3.apply_mpo(ops.plus[axis], origin; tol = 0.0, cap = 8)) == 0
    end
    signed = Q3.cross(
        rec,
        9,
        j -> begin
            q = ntuple(a -> sum(((j >> (3b + a - 1)) & 1) << b for b = 0:2), 3)
            exp(sum((0.1a) * (q[a] < 4 ? q[a] : q[a] - 8) for a = 1:3))
        end;
        tol = 1e-12,
        cap = 32,
    )
    kernel = K.cartesian_toeplitz(signed, 2)
    source = (3, 0, 2)
    response =
        values3(Q3.apply_mpo(kernel, Q3.delta_tt(6, grid_index(g, source)); tol = 1e-12, cap = 32))
    reference = [exp(sum(0.1a * (grid_indices(g, j)[a] - source[a]) for a = 1:3)) for j = 0:63]
    @test norm(response - reference) < 1e-10
    green = K.green_operator(p, a, SCFOptions(), phys, ops)
    y = Q3.apply_mpo(green, psi; tol = 1e-12, cap = 32)
    relative = norm((T + 0.5I) * values3(y) - densepsi)
    @test relative < 2e-9
    # Acceptance must see error discarded by a restrictive storage rank.
    restricted = Q3.rounded(y; tol = 0.0, cap = 1)
    residual = K.green_residual(green, restricted, psi)
    residual_norm = sqrt(real(only(Q3.expression_gram([residual]))))
    @test residual_norm ≈ norm(densepsi - (T + 0.5I) * values3(restricted)) rtol = 1e-10
    @test residual_norm > 1e-6
    limited_green = K.CartesianGreen(green.kinetic, green.shift, 1e-9, 20, 32)
    @test_throws ErrorException Q3.apply_mpo(limited_green, psi; tol = 1e-12, cap = 1)

    # Local-potential reuse must respect its own field budget.
    low_cap = Accuracy(field_cap = 32, effective_potential_cap = 1)
    limited_H = K.effective_hamiltonian(p, rho, ops, low_cap, rec)
    @test Q3.rank(limited_H.potential) == 1
    println("3D Hamiltonian action error = ", norm(action - Hd * densepsi))
    println("3D discrete Green relative residual = ", relative)
end

@testset "Isolated Hartree, SCF, checkpoints, and configuration" begin
    g = BoxGrid3D(2, 8.0)
    p = Problem(g, AllElectronLDA3D([(0.0, 0.0, 0.0)], [2.0]), [2.0])
    a = Accuracy(
        orbital_cap = 16,
        field_cap = 32,
        work_cap = 32,
        work_tolerance = 1e-9,
        storage_tolerance = 1e-11,
    )
    rec = K.recorder(nothing, TCIOptions())
    ops = K.build_operators(p, a, rec, PhysicsNumerics())
    xyz = [grid_position(g, j) for j = 0:(length(g) - 1)]
    C = [K.coulomb_cell_average(x .- y, K.spacing(g)) for x in xyz, y in xyz]
    for index in (0, 7, 21, length(g) - 1)
        point = Q3.delta_tt(nsites(g), index)
        convolution = values3(Q3.apply_mpo(ops.interaction, point; tol = 1e-12, cap = 32))
        @test norm(convolution - C[:, index + 1]) < 1e-9
    end
    T = dense_kinetic3(g)
    ext = values3(ops.external) |> real
    weights = fill(2.0 / length(g), length(g))
    dense_energy = Inf
    for iteration = 1:500
        rho = weights / volume_element(g)
        vh = C * weights
        vx = last.(K.lda_pz81.(rho))
        f = eigen(Symmetric(T + Diagonal(ext + vh + vx)))
        output = 2abs2.(f.vectors[:, 1])
        if norm(output - weights) < 1e-12
            weights = output
            orbital = f.vectors[:, 1]
            rho = weights / volume_element(g)
            dense_energy =
                2dot(orbital, T * orbital) +
                dot(weights, ext) +
                dot(weights, C * weights) / 2 +
                volume_element(g) * sum(first.(K.lda_pz81.(rho)))
            break
        end
        weights = 0.5weights + 0.5output
    end
    @test isfinite(dense_energy)
    mktempdir() do dir
        result = ground_state(
            p;
            accuracy = a,
            scf = SCFOptions(
                criterion = :strict,
                max_iterations = 50,
                energy_tolerance = 1e-8,
                eigen_tolerance = 1e-6,
                density_tolerance = 1e-6,
                mixing = 0.5,
            ),
            output = joinpath(dir, "scf"),
        )
        @test result.converged
        @test result.audit.passed
        @test abs(result.audit.energy.total - dense_energy) < 1e-7
        restored = load_checkpoint(joinpath(dir, "scf", "checkpoint.jls"); problem = p)
        @test restored.problem.grid == g
        @test values3(restored.orbitals[1]) ≈ values3(result.state.orbitals[1])
        wrong = Problem(BoxGrid3D(2, 9.0), p.model, p.occupations)
        @test_throws ArgumentError load_checkpoint(
            joinpath(dir, "scf", "checkpoint.jls");
            problem = wrong,
        )
        @test_throws ArgumentError propagate(result.state; time = TimeOptions(steps = 1))
        measurement = observe(restored; accuracy = a)
        @test measurement.electron_number ≈ 2 atol = 1e-10
        refined_problem = Problem(BoxGrid3D(3, 8.0), p.model, p.occupations)
        refined = refine_state(
            restored,
            refined_problem;
            accuracy = a,
            refinement = RefinementOptions(degree = 1),
        )
        @test length(refined.orbitals[1]) == 9
        @test Q3.ttnorm(refined.orbitals[1]) ≈ 1 atol = 1e-10
        out = postprocess(joinpath(dir, "scf"); accuracy = a, samples = 16)
        @test first(readlines(joinpath(out, "density.csv"))) == "x,y,z,density"
        println("3D SCF energy = ", result.audit.energy.total)
        println("3D dense SCF energy difference = ", abs(result.audit.energy.total - dense_energy))
        println("3D SCF audited residual = ", result.audit.max_residual)
        println("3D SCF audited density residual = ", result.audit.density_residual)
    end
    for file in ("h12-3d.toml", "he-3d-smoke.toml")
        plan = configuration_plan(joinpath(@__DIR__, "..", "inputs", file))
        @test plan["spatial_dimensions"] == 3
        @test !plan["calculation_started"]
    end
    cluster = hydrogen_cluster(R = 2, shape = (2, 2, 1), cell = (12, 12, 8))
    result = ground_state(cluster; accuracy = a, scf = SCFOptions(max_iterations = 3))
    @test isfinite(result.audit.energy.total)
    @test result.audit.gram_defect < 1e-8
    @test result.audit.electron_number ≈ 4 atol = 1e-9
    @test_throws ArgumentError fixed_spectrum(result.state, zeros(length(cluster.grid)))
    @test_throws ArgumentError lowpass_mpo(cluster; cutoff = 1.0)
end
