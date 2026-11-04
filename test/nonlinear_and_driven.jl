module NonlinearAndDrivenChecks

using Test, QuanticsKS, LinearAlgebra, Random
const K = QuanticsKS
const Q = K.Q
BLAS.set_num_threads(1)
Random.seed!(314)
const a = Accuracy(
    field_tolerance = 1e-12,
    algebra_tolerance = 1e-14,
    density_tolerance = 1e-12,
    external_tolerance = 1e-13,
    kernel_tolerance = 1e-13,
    orthogonalization_tolerance = 1e-13,
    orbital_cap = 32,
    field_cap = 32,
    work_cap = 64,
)
function kinetic(p)
    N = length(p.grid)
    h = K.spacing(p.grid)
    T = Matrix{ComplexF64}(I, N, N) / (p.mass * h^2)
    for j = 1:N
        T[j, mod1(j + 1, N)] = -1 / (2p.mass * h^2)
        T[j, mod1(j - 1, N)] = -1 / (2p.mass * h^2)
    end
    T
end
values(s) =
    hcat([[Q.Evaluator(x)(j) for j = 0:(length(s.problem.grid) - 1)] for x in s.orbitals]...)
settings(dt, n) = TimeOptions(
    dt = dt,
    steps = n,
    phase_tolerance = 1e-13,
    kinetic_tolerance = 1e-14,
    output_tolerance = 1e-14,
    phase_cap = 32,
    kinetic_cap = 64,
    output_cap = 32,
    observe_every = n,
    snapshot_every = n,
)
@testset "Nonlinear density feedback and occupied-space preservation" begin
    p = hydrogen_chain(R = 5, natoms = 4, distance = 4.0)
    s = state_from_functions(
        p,
        [x -> exp(0.2cos(2pi * x / p.grid.L)), x -> cis(2pi * x / p.grid.L)];
        accuracy = a,
    )
    h = K.spacing(p.grid)
    N = length(p.grid)
    u = values(s)
    T = kinetic(p)
    dt = 0.015
    tab = Q.XCTable()
    positions = (0:(N - 1)) .* h
    W = [K.kernel_value(p.model, x - y, p.grid.L) for x in positions, y in positions]
    ext = K.external_value.(Ref(p.model), positions, p.grid.L)
    function veff(u)
        n = vec(sum(abs2.(u) .* permutedims(p.occupations); dims = 2)) / h
        ext + h * W * n + [Q.xc(tab, v)[2] for v in n]
    end
    v = veff(u)
    u1 = exp.(-im * dt * v / 2) .* u
    u2 = (I + im * dt * T / 2) \ ((I - im * dt * T / 2) * u1)
    exact = exp.(-im * dt * veff(u2) / 2) .* u2
    result = propagate(s; accuracy = a, time = settings(dt, 1))
    @test norm(values(result.state) - exact) < 2e-10
    @test norm(values(result.state)' * values(result.state) - I) < 2e-10
    @test last(result.observations).electron_number ≈ 4 atol = 2e-10
end
@testset "Time-dependent pulse against dense RK4" begin
    p = Problem(PeriodicGrid1D(5, 8), LocalPotential1D(0.0), [1.0])
    s = state_from_functions(p, [x -> cis(2pi * x / 8)]; accuracy = a)
    pulse = CosinePulse(
        amplitude = 0.3,
        duration = 0.4,
        frequency = 2.0,
        spatial_mode = 2,
        phase = 0.2,
        spatial_phase = 0.1,
    )
    T = kinetic(p)
    grid = (0:31) .* K.spacing(p.grid)
    rhs(u, t) = -im * (T * u + K.drive_value.(Ref(pulse), grid, t, p.grid.L) .* u)
    u = vec(values(s))
    dr = 0.4 / 2048
    for j = 0:2047
        t = j * dr
        k1 = rhs(u, t)
        k2 = rhs(u + dr * k1 / 2, t + dr / 2)
        k3 = rhs(u + dr * k2 / 2, t + dr / 2)
        k4 = rhs(u + dr * k3, t + dr)
        u += dr * (k1 + 2k2 + 2k3 + k4) / 6
    end
    errors = Float64[]
    for (dt, n) in ((0.025, 16), (0.0125, 32))
        result = propagate(s; accuracy = a, drive = pulse, time = settings(dt, n))
        push!(errors, norm(vec(values(result.state)) - u))
    end
    @test errors[2] < 1e-5
    @test 3.5 < errors[1] / errors[2] < 4.5
    println("Driven errors against dense RK4: ", errors)
end

end
