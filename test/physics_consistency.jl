module PhysicsConsistencyChecks
using Test, QuanticsKS, LinearAlgebra, Random, QuadGK, SpecialFunctions, JSON3
const K = QuanticsKS
const Q = K.Q
const measurements = Dict{String,Any}()
BLAS.set_num_threads(1)

# Independent image sum, extrapolated in the number of images; no Legendre series.
function periodic_kernel(x, L; softening = 1.0, coupling = 1.0)
    x = mod(x + L / 2, L) - L / 2
    function images(count)
        inv(hypot(x, softening)) + sum(
            inv(hypot(x + k * L, softening)) + inv(hypot(x - k * L, softening)) - 2 / (k * L) for
            k = 1:count
        )
    end
    coarse, fine = images(2048), images(4096)
    ratio = ((4096 + 0.5) / (2048 + 0.5))^2
    coupling * (ratio * fine - coarse) / (ratio - 1)
end

function correlation_energy(rho)
    rs = 1 / (2rho)
    -rho / 2 * (rs + 0.012827rs^2) / (18.40 + 7.501rs^2 + 0.10185rs^3) *
    log1p(1.511rs + 0.258rs^4.424)
end

# Direct dense exchange quadrature after u=t^2; correlation uses a complex-step derivative.
function dense_xc(rho)
    rho = max(rho, 1e-12)
    exchange_energy = -rho^2 * quadgk(0.0, 1.0; rtol = 1e-12) do t
        iszero(t) ? 0.0 : 2t * (1 - t^2) * besselk(0, pi * rho * t^2)
    end[1]
    exchange_potential = -rho * quadgk(0.0, 1.0; rtol = 1e-12) do t
        iszero(t) ? 0.0 : 2t * besselk(0, pi * rho * t^2)
    end[1]
    step = rho * 1e-20
    Vcorrelation = imag(correlation_energy(rho + im * step)) / step
    exchange_energy + correlation_energy(rho), exchange_potential + Vcorrelation
end

values(psi) =
    [Q.TCI.evaluate(Q.tensortrain(psi), Q.bits(j, length(psi))) for j = 0:(2^length(psi) - 1)]

@testset "XC energy and potential against independent dense quadrature" begin
    table = Q.XCTable()
    densities = exp.(range(log(1e-12), log(8.0); length = 121))
    references = dense_xc.(densities)
    exact = Q.xc_exact.(densities)
    interpolated = Q.xc.(Ref(table), densities)
    energy_error = maximum(abs(first(a) - first(b)) for (a, b) in zip(interpolated, references))
    potential_error = maximum(abs(last(a) - last(b)) for (a, b) in zip(interpolated, references))
    @test maximum(abs(first(a) - first(b)) for (a, b) in zip(exact, references)) < 2e-11
    @test maximum(abs(last(a) - last(b)) for (a, b) in zip(exact, references)) < 2e-11
    @test energy_error < 2e-9
    @test potential_error < 2e-8
    derivative_error = maximum(exp.(range(log(2e-8), log(3.9); length = 43))) do rho
        step = rho * 1e-5
        derivative = (first(Q.xc(table, rho + step)) - first(Q.xc(table, rho - step))) / (2step)
        abs(derivative - last(Q.xc(table, rho)))
    end
    @test derivative_error < 2e-8
    measurements["xc"] = (;
        energy_density_max_error = energy_error,
        potential_max_error_Ha = potential_error,
        energy_derivative_max_error = derivative_error,
    )
end

@testset "Full 1D Hamiltonian and energy against independent dense fields" begin
    Random.seed!(20261106)
    grid = PeriodicGrid1D(6, 8.0)
    dx, ngrid = K.spacing(grid), length(grid)
    accuracy = Accuracy(
        field_tolerance = 1e-12,
        density_tolerance = 1e-12,
        external_tolerance = 1e-13,
        kernel_tolerance = 1e-13,
        algebra_tolerance = 1e-13,
        xc_energy_tolerance = 1e-12,
        orbital_cap = 32,
        work_cap = 32,
        field_cap = 32,
    )
    models = (
        SoftCoulombLSDA1D([0.31, 3.27], [0.7, 1.3]),
        SoftCoulombLSDA1D([0.31, 3.27], [0.7, 1.3]; softening = 0.6, coupling = 1.2, xc = :none),
        LocalPotential1D(x -> 0.3cos(2pi * x / grid.L + 0.2); id = "dense-audit-cosine"),
    )
    for (case, model) in enumerate(models)
        mass = case == 1 ? 1.0 : 1.7
        problem = Problem(grid, model, [2.0]; mass)
        rec = K.recorder(nothing, TCIOptions())
        ops = K.build_operators(problem, accuracy, rec, PhysicsNumerics())
        psi = Q.normalize(
            Q.cross(
                rec,
                grid.R,
                j -> 1 + 0.21cis(2pi * j / ngrid) + 0.13im * cis(6pi * j / ngrid);
                tol = 1e-13,
                cap = 32,
            ),
        )
        psi_dense = values(psi)
        rho_dense = 2abs2.(psi_dense) / dx
        rho = K.density(problem, [psi], accuracy, rec)
        interacting = model isa SoftCoulombLSDA1D
        kernel(x) =
            periodic_kernel(x, grid.L; softening = model.softening, coupling = model.coupling)
        Vext =
            interacting ?
            [
                -sum(
                    charge * kernel(j * dx - position) for
                    (position, charge) in zip(model.positions, model.charges)
                ) for j = 0:(ngrid - 1)
            ] : [0.3cos(2pi * j / ngrid + 0.2) for j = 0:(ngrid - 1)]
        convolution = interacting ? [kernel(j * dx) for j = 0:(ngrid - 1)] : zeros(ngrid)
        Vhartree =
            dx * [
                sum(convolution[mod(i - j, ngrid) + 1] * rho_dense[j + 1] for j = 0:(ngrid - 1)) for
                i = 0:(ngrid - 1)
            ]
        xc = case == 1 ? dense_xc.(rho_dense) : fill((0.0, 0.0), ngrid)
        Vxc = last.(xc)
        drive = FunctionDrive((x, t) -> 0.07sin(2pi * x / grid.L) * cos(t), "dense-audit-drive")
        time = 0.37
        Vdrive = [0.07sin(2pi * j / ngrid) * cos(time) for j = 0:(ngrid - 1)]
        Veff = Vext + Vhartree + Vxc + Vdrive
        H = K.effective_hamiltonian(problem, rho, ops, accuracy, rec; time, drive)
        shift = circshift(Matrix{Float64}(I, ngrid, ngrid), (1, 0))
        kinetic = (2I - shift - shift') / (2mass * dx^2)
        Hdense = kinetic + Diagonal(Veff)
        Hpsi = Q.h_action(H, psi, K.solver_config(problem, accuracy), rec; work_tol = 1e-12)
        nuclear =
            interacting ?
            model.charges[1] * model.charges[2] * kernel(model.positions[1] - model.positions[2]) :
            0.0
        expected_energy = (;
            kinetic = 2real(dot(psi_dense, kinetic * psi_dense)),
            external = dx * dot(rho_dense, Vext),
            hartree = dx / 2 * dot(rho_dense, Vhartree),
            xc = dx * sum(first, xc),
            nuclear,
            drive = dx * dot(rho_dense, Vdrive),
        )
        energy = K.total_energy(problem, [psi], rho, ops, accuracy, rec; time, drive)
        energy_errors = Dict(
            string(key) => abs(getproperty(energy, key) - value) for
            (key, value) in pairs(expected_energy)
        )
        energy_errors["total"] = abs(energy.total - sum(expected_energy))
        field_error = maximum(abs, values(H.potential) - Veff)
        action_error = norm(values(Hpsi) - Hdense * psi_dense)
        @test maximum(abs, values(rho) - rho_dense) < 2e-10
        @test maximum(abs, values(ops.external) - Vext) < 2e-10
        @test maximum(abs, values(K.hartree(rho, problem, ops, accuracy)) - Vhartree) < 2e-10
        @test field_error < 2e-8
        @test action_error < 2e-8
        @test maximum(Base.values(energy_errors)) < 2e-8
        measurements["case_$case"] = (;
            field_max_error_Ha = field_error,
            hamiltonian_action_l2_error = action_error,
            energy_errors_Ha = energy_errors,
        )
        close(rec)
    end
end

if haskey(ENV, "QUANTICSKS_PHYSICS_REPORT")
    open(io -> JSON3.pretty(io, measurements), ENV["QUANTICSKS_PHYSICS_REPORT"], "w")
end
end
