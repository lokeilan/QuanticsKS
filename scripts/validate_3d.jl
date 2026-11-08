# Preserve the input, source snapshot, checkpoint, raw logs, and measured report.
using QuanticsKS, LinearAlgebra, TOML, JSON3
const K = QuanticsKS
const Q = K.Q
BLAS.set_num_threads(1)

function dense_reference(p; tolerance = 1e-11, iterations = 500)
    g = p.grid
    length(g) <= 512 || error("Dense validation is limited to at most 512 grid points")
    n, h, dv = 1 << g.R, K.spacing(g), volume_element(g)
    T = zeros(length(g), length(g))
    for j = 0:(length(g) - 1)
        q = grid_indices(g, j)
        for a = 1:3
            T[j + 1, j + 1] += 1 / (p.mass * h[a]^2)
            for sign in (-1, 1)
                neighbor = ntuple(b -> q[b] + (a == b ? sign : 0), 3)
                all(v -> 0 <= v < n, neighbor) || continue
                T[j + 1, grid_index(g, neighbor) + 1] -= 1 / (2p.mass * h[a]^2)
            end
        end
    end
    # Share the independently quadrature-checked scalar physics, not QTT operators.
    positions = [grid_position(g, j) for j = 0:(length(g) - 1)]
    kernel_cache = Dict{NTuple{3,Int},Float64}()
    C = [
        get!(kernel_cache, Tuple(grid_indices(g, i) .- grid_indices(g, j))) do
            K.coulomb_cell_average(positions[i + 1] .- positions[j + 1], h)
        end for i = 0:(length(g) - 1), j = 0:(length(g) - 1)
    ]
    external = [K.external_value(p.model, r, g) for r in positions]
    b = length(p.occupations)
    weights = fill(sum(p.occupations) / length(g), length(g))
    for iteration = 1:iterations
        rho = weights / dv
        vx = p.model.xc == :none ? zero(rho) : last.(K.lda_pz81.(rho))
        H = T + Diagonal(external + C * weights + vx)
        f = eigen(Symmetric(H), 1:b)
        U = f.vectors
        next = abs2.(U) * p.occupations
        if norm(next - weights) < tolerance
            rho = next / dv
            exc = p.model.xc == :none ? 0.0 : dv * sum(first.(K.lda_pz81.(rho)))
            kinetic = sum(p.occupations[i] * dot(U[:, i], T * U[:, i]) for i = 1:b)
            energy =
                kinetic +
                dot(next, external) +
                dot(next, C * next) / 2 +
                exc +
                K.nuclear_energy(p.model, g)
            return (; energy, iterations = iteration, density = rho)
        end
        weights = (weights + next) / 2
    end
    error("Dense reference did not converge")
end

function main(args)
    length(args) in (1, 2) || error("Usage: validate_3d.jl NEW_OUTPUT [R=3]")
    output = abspath(args[1])
    ispath(output) && error("Choose a new validation output directory")
    R = length(args) == 2 ? parse(Int, args[2]) : 3
    R in (2, 3) || error("Validation R must be 2 or 3")
    mkpath(output)
    settings = TOML.parsefile(joinpath(@__DIR__, "..", "inputs", "he-3d-smoke.toml"))
    settings["grid"]["R"] = R
    settings["accuracy"]["orbital_cap"] = 32
    settings["accuracy"]["field_cap"] = 64
    settings["accuracy"]["work_cap"] = 64
    input = joinpath(output, "input.toml")
    open(io -> TOML.print(io, settings), input, "w")
    result = run_config(input; output = joinpath(output, "run"))
    reference = dense_reference(result.state.problem)
    rho = K.density(
        result.state.problem,
        result.state.orbitals,
        K.options_from_dict(Accuracy, settings["accuracy"]),
        K.recorder(nothing, TCIOptions()),
    )
    ev = Q.Evaluator(rho)
    density = real.([ev(j) for j = 0:(length(result.state.problem.grid) - 1)])
    measurements = (;
        R,
        qtt_energy = result.audit.energy.total,
        dense_energy = reference.energy,
        energy_error = abs(result.audit.energy.total - reference.energy),
        density_relative_error = norm(density - reference.density) / norm(reference.density),
        orbital_residual = result.audit.max_residual,
        density_residual = result.audit.density_residual,
        electron_number = result.audit.electron_number,
        gram_defect = result.audit.gram_defect,
        stopping_criterion = settings["scf"]["criterion"],
        stop_reason = string(result.stop_reason),
        solver_converged = result.converged,
        physical_acceptance = result.audit.passed,
        scf_iterations = length(result.history),
        elapsed_seconds = result.elapsed_seconds,
        itensormps_version = string(pkgversion(Q.ITensorMPS)),
        itensors_version = string(pkgversion(Q.ITensors)),
    )
    open(io -> JSON3.pretty(io, measurements), joinpath(output, "measurements.json"), "w")
    open(joinpath(output, "validation.md"), "w") do io
        println(io, "# 3D QTT discrete-model validation\n")
        println(
            io,
            "Cell-centered isolated helium, R=$R bits per axis, box length 8 bohr. The dense reference solves the same discrete model independently of the QTT operators. Scalar Coulomb and XC formulas are shared and are separately checked by quadrature and differentiation in the regression.\n",
        )
        println(io, "| Measurement | Value |\n|---|---:|")
        for (name, value) in pairs(measurements)
            println(io, "| $name | $value |")
        end
        println(
            io,
            "\nThese results validate implementation consistency on this grid. They do not establish continuum accuracy, box convergence, the paper's extrapolated energies, or fine-grid performance. Energy stopping and physical acceptance are recorded separately.",
        )
    end
    println(JSON3.write(measurements))
    measurements.physical_acceptance &&
        measurements.energy_error < 1e-7 &&
        measurements.density_relative_error < 1e-5 || error("3D discrete-model validation failed")
end
main(ARGS)
