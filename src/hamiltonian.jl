# Build QTT operators from the model and density, then evaluate density and total energy.

# The two concrete models share grid operators and solvers. Model-specific
# interactions live here, not in the SCF or propagation loops.
kernel_value(model::SoftCoulombLSDA1D, x, L) =
    model.coupling * Q.soft(x / model.softening, L / model.softening) / model.softening
external_value(model::SoftCoulombLSDA1D, x, L) =
    -sum(
        charge * kernel_value(model, x - position, L) for
        (position, charge) in zip(model.positions, model.charges)
    )
function external_value(model::LocalPotential1D, x, L)
    value = model.potential(x)
    isreal(value) && isfinite(value) ||
        throw(ArgumentError("A finite real local potential is required"))
    real(value)
end
nuclear_energy(model::LocalPotential1D, L) = 0.0
nuclear_energy(model::SoftCoulombLSDA1D, L) = sum(
    (
        model.charges[i] *
        model.charges[j] *
        kernel_value(model, model.positions[i] - model.positions[j], L) for
        i in eachindex(model.positions) for j = 1:(i - 1)
    );
    init = 0.0,
)
has_xc(model) = model isa SoftCoulombLSDA1D && model.xc == :lsda

struct Operators{W,T,S}
    external::Q.QTT
    interaction::W
    xc_table::T
    plus::Vector{Array{ComplexF64,4}}
    minus::Vector{Array{ComplexF64,4}}
    external_samples::S
    nuclear::Float64
end
function build_operators(problem, accuracy, rec, physics::PhysicsNumerics)
    validate(physics)
    grid = problem.grid
    dx = spacing(grid)
    ngrid = length(grid)
    pivots =
        problem.model isa SoftCoulombLSDA1D ?
        round.(Int, mod.(problem.model.positions, grid.L) ./ dx) .% ngrid : Int[]
    model = deepcopy(problem.model)
    external_samples =
        Q.memoized_sample(Q.sampled(j -> external_value(model, j * dx, grid.L), grid.R), rec)
    built = Q.recorded_map(Any, 2, rec.options, rec) do item, localrec
        if item == 1
            problem.model isa LocalPotential1D && problem.model.potential isa ConstantPotential ?
            Q.constant_tt(grid.R, problem.model.potential.value) :
            Q.cross(
                localrec,
                grid.R,
                external_samples;
                tol = accuracy.external_tolerance,
                cap = field_caps(accuracy).external_cap,
                label = "external",
                pivots,
            )
        else
            problem.model isa SoftCoulombLSDA1D ?
            Q.circulant(
                Q.cross(
                    localrec,
                    grid.R,
                    j -> kernel_value(problem.model, j * dx, grid.L);
                    tol = accuracy.kernel_tolerance,
                    cap = field_caps(accuracy).kernel_cap,
                    label = "interaction",
                ),
            ) : nothing
        end
    end
    external, interaction = built
    xc_table =
        has_xc(problem.model) ?
        Q.XCTable(
            npoints = physics.xc_table_points,
            density_min = physics.xc_density_min,
            density_max = physics.xc_density_max,
            rtol = physics.exchange_quadrature_rtol,
            parallel_tasks = get(rec.options, :parallel_tasks, 1),
        ) : nothing
    Operators(
        external,
        interaction,
        xc_table,
        Q.shift_mpo(grid.R, 1),
        Q.shift_mpo(grid.R, -1),
        external_samples,
        nuclear_energy(model, grid.L),
    )
end
# Cores store sqrt(dV)*psi, so divide by the volume element to recover density.
function density(problem, orbitals, accuracy, rec; renormalize = false, label = "density")
    dx = volume_element(problem.grid)
    orbital_samples = Q.block_oracle(orbitals, rec)
    rho = Q.cross(
        rec,
        nsites(problem.grid),
        Q.DensitySamples(orbital_samples, Float64.(problem.occupations ./ dx));
        tol = accuracy.density_tolerance,
        cap = field_caps(accuracy).density_cap,
        label,
    )
    if renormalize
        charge = dx * real(sum(Q.tensortrain(rho)))
        charge > 0 || error("Nonpositive density integral")
        return Q.scale(rho, sum(problem.occupations) / charge)
    end
    rho
end
# Multiply the cyclic convolution by dx once to obtain the spatial integral.
hartree(rho, problem, ops, accuracy) =
    ops.interaction === nothing ? Q.constant_tt(problem.grid.R, 0) :
    Q.scale(
        Q.apply_mpo(
            ops.interaction,
            rho;
            tol = accuracy.algebra_tolerance,
            cap = field_caps(accuracy).hartree_cap,
        ),
        spacing(problem.grid),
    )
"Freeze one density and share its Hartree/XC expression across local consumers."
function local_potential_samples(problem, rho, ops, accuracy, rec; time = 0.0, drive = NoDrive())
    if problem.model isa LocalPotential1D && drive isa NoDrive
        return ops.external_samples
    end
    # Hartree and XC read the same immutable SCF input, never updated orbitals.
    frozen_density = Q.QTT([copy(core) for core in rho.c])
    Vhartree = hartree(frozen_density, problem, ops, accuracy)
    rho_samples = Q.memoized_sample(frozen_density, rec)
    hartree_samples = Q.memoized_sample(Vhartree, rec)
    dx = spacing(problem.grid)
    Q.memoized_sample(
        Q.pointwise(
            (external, hartree, density, drive) ->
                real(external + hartree) +
                (ops.xc_table === nothing ? 0.0 : Q.xc(ops.xc_table, real(density))[2]) +
                drive,
            ops.external_samples,
            hartree_samples,
            rho_samples,
            Q.sampled(j -> drive_value(drive, j * dx, time, problem.grid.L), problem.grid.R),
        ),
        rec,
    )
end

function effective_hamiltonian(problem, rho, ops, accuracy, rec; time = 0.0, drive = NoDrive())
    field = local_potential_samples(problem, rho, ops, accuracy, rec; time, drive)
    # Core projections need one shared field representation; sampled residuals
    # and Green sources consume the original local expression directly.
    Veff =
        problem.model isa LocalPotential1D && drive isa NoDrive &&
        Q.rank(ops.external) <= field_caps(accuracy).effective_potential_cap ? ops.external :
        Q.cross(
            rec,
            problem.grid.R,
            field;
            tol = accuracy.field_tolerance,
            cap = field_caps(accuracy).effective_potential_cap,
            label = "effective_potential",
        )
    Q.Hamiltonian(
        Veff,
        Q.diagonal(Veff),
        ops.plus,
        ops.minus,
        spacing(problem.grid) * sqrt(problem.mass),
        field,
    )
end
function total_energy(problem, orbitals, rho, ops, accuracy, rec; time = 0.0, drive = NoDrive())
    dx = spacing(problem.grid)
    kinetic_terms =
        Q.recorded_map(Float64, length(orbitals), rec.options, rec; random = false) do i, localrec
            occupation, psi = problem.occupations[i], orbitals[i]
            shifted = Q.apply_mpo(
                ops.plus,
                psi;
                tol = accuracy.algebra_tolerance,
                cap = accuracy.orbital_cap,
            )
            gradient_norm =
                Q.difference_norm(shifted, psi; rec = localrec, label = "kinetic_difference_norm")
            occupation * gradient_norm^2 / (2problem.mass * dx^2)
        end
    kinetic_energy = sum(kinetic_terms)
    Vhartree = hartree(rho, problem, ops, accuracy)
    external_energy = dx * real(Q.inner(rho, ops.external; cfg = rec.options))
    hartree_energy = 0.5dx * real(Q.inner(rho, Vhartree; cfg = rec.options))
    xc_energy = 0.0
    if ops.xc_table !== nothing
        rho_samples = Q.sampled(rho, rec)
        e = Q.cross(
            rec,
            problem.grid.R,
            Q.pointwise(value -> Q.xc(ops.xc_table, real(value))[1], rho_samples);
            tol = accuracy.xc_energy_tolerance,
            cap = field_caps(accuracy).xc_energy_cap,
            label = "xc_energy_density",
        )
        xc_energy = dx * real(sum(Q.tensortrain(e)))
    end
    drive_energy = 0.0
    if !(drive isa NoDrive)
        Veff = Q.cross(
            rec,
            problem.grid.R,
            j -> drive_value(drive, j * dx, time, problem.grid.L);
            tol = accuracy.field_tolerance,
            cap = field_caps(accuracy).drive_cap,
            label = "drive_energy",
        )
        drive_energy = dx * real(Q.inner(rho, Veff; cfg = rec.options))
    end
    ion_energy = ops.nuclear
    (;
        total = kinetic_energy +
                external_energy +
                hartree_energy +
                xc_energy +
                ion_energy +
                drive_energy,
        kinetic = kinetic_energy,
        external = external_energy,
        hartree = hartree_energy,
        xc = xc_energy,
        nuclear = ion_energy,
        drive = drive_energy,
    )
end
