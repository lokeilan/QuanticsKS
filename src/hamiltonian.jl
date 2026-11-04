# Build QTT operators from the model and density, then evaluate density and total energy.

# The two concrete models share grid operators and solvers. Model-specific
# interactions live here, not in the SCF or propagation loops.
kernel_value(m::SoftCoulombLSDA1D, x, L) =
    m.coupling * Q.soft(x / m.softening, L / m.softening) / m.softening
external_value(m::SoftCoulombLSDA1D, x, L) =
    -sum(z * kernel_value(m, x - a, L) for (a, z) in zip(m.positions, m.charges))
function external_value(m::LocalPotential1D, x, L)
    value = m.potential(x)
    isreal(value) && isfinite(value) ||
        throw(ArgumentError("A finite real local potential is required"))
    real(value)
end
nuclear_energy(m::LocalPotential1D, L) = 0.0
nuclear_energy(m::SoftCoulombLSDA1D, L) = sum(
    (
        m.charges[a] * m.charges[b] * kernel_value(m, m.positions[a] - m.positions[b], L) for
        a in eachindex(m.positions) for b = 1:(a - 1)
    );
    init = 0.0,
)
has_xc(m) = m isa SoftCoulombLSDA1D && m.xc == :lsda

struct Operators{W,T,S}
    external::Q.QTT
    interaction::W
    xc_table::T
    plus::Vector{Array{ComplexF64,4}}
    minus::Vector{Array{ComplexF64,4}}
    external_samples::S
    nuclear::Float64
end
function build_operators(p, a, rec, physics::PhysicsNumerics)
    validate(physics)
    g = p.grid
    h = spacing(g)
    N = length(g)
    pivots =
        p.model isa SoftCoulombLSDA1D ? round.(Int, mod.(p.model.positions, g.L) ./ h) .% N : Int[]
    model = deepcopy(p.model)
    external_samples =
        Q.memoized_sample(Q.sampled(j -> external_value(model, j * h, g.L), g.R), rec)
    built = Q.recorded_map(Any, 2, rec.options, rec) do item, localrec
        if item == 1
            p.model isa LocalPotential1D && p.model.potential isa ConstantPotential ?
            Q.constant_tt(g.R, p.model.potential.value) :
            Q.cross(
                localrec,
                g.R,
                external_samples;
                tol = a.external_tolerance,
                cap = a.field_cap,
                label = "external",
                pivots,
            )
        else
            p.model isa SoftCoulombLSDA1D ?
            Q.circulant(
                Q.cross(
                    localrec,
                    g.R,
                    j -> kernel_value(p.model, j * h, g.L);
                    tol = a.kernel_tolerance,
                    cap = a.field_cap,
                    label = "interaction",
                ),
            ) : nothing
        end
    end
    external, w = built
    tab =
        has_xc(p.model) ?
        Q.XCTable(
            npoints = physics.xc_table_points,
            density_min = physics.xc_density_min,
            density_max = physics.xc_density_max,
            rtol = physics.exchange_quadrature_rtol,
            parallel_tasks = get(rec.options, :parallel_tasks, 1),
        ) : nothing
    Operators(
        external,
        w,
        tab,
        Q.shift_mpo(g.R, 1),
        Q.shift_mpo(g.R, -1),
        external_samples,
        nuclear_energy(model, g.L),
    )
end
# Cores store sqrt(h)*psi, so divide by h to recover physical density.
function density(p, xs, a, rec; renormalize = false, label = "density")
    h = spacing(p.grid)
    o = Q.block_oracle(xs, rec)
    n = Q.cross(
        rec,
        p.grid.R,
        Q.DensitySamples(o, Float64.(p.occupations ./ h));
        tol = a.density_tolerance,
        cap = a.field_cap,
        label,
    )
    if renormalize
        charge = h * real(Q.inner(Q.constant_tt(p.grid.R), n; cfg = rec.options))
        charge > 0 || error("Nonpositive density integral")
        return Q.scale(n, sum(p.occupations) / charge)
    end
    n
end
# Multiply the cyclic convolution by h once to obtain the spatial integral.
hartree(n, p, ops, a) =
    ops.interaction === nothing ? Q.constant_tt(p.grid.R, 0) :
    Q.scale(
        Q.apply_mpo(ops.interaction, n; tol = a.algebra_tolerance, cap = a.field_cap),
        spacing(p.grid),
    )
"Freeze one density and share its Hartree/XC expression across local consumers."
function local_potential_samples(p, n, ops, a, rec; time = 0.0, drive = NoDrive())
    if p.model isa LocalPotential1D && drive isa NoDrive
        return ops.external_samples
    end
    # Hartree and XC read the same immutable SCF input, never updated orbitals.
    frozen_density = Q.QTT([copy(g) for g in n.c])
    vh = hartree(frozen_density, p, ops, a)
    en = Q.memoized_sample(frozen_density, rec)
    eh = Q.memoized_sample(vh, rec)
    h = spacing(p.grid)
    Q.memoized_sample(
        Q.pointwise(
            (external, hartree, density, drive) ->
                real(external + hartree) +
                (ops.xc_table === nothing ? 0.0 : Q.xc(ops.xc_table, real(density))[2]) +
                drive,
            ops.external_samples,
            eh,
            en,
            Q.sampled(j -> drive_value(drive, j * h, time, p.grid.L), p.grid.R),
        ),
        rec,
    )
end

function effective_hamiltonian(p, n, ops, a, rec; time = 0.0, drive = NoDrive())
    field = local_potential_samples(p, n, ops, a, rec; time, drive)
    if p.model isa LocalPotential1D && drive isa NoDrive
        v = ops.external
        return Q.Hamiltonian(
            v,
            Q.diagonal(v),
            ops.plus,
            ops.minus,
            spacing(p.grid) * sqrt(p.mass),
            field,
        )
    end
    # Core projections need one shared field representation; sampled residuals
    # and Green sources consume the original local expression directly.
    v = Q.cross(
        rec,
        p.grid.R,
        field;
        tol = a.field_tolerance,
        cap = a.field_cap,
        label = "effective_potential",
    )
    Q.Hamiltonian(v, Q.diagonal(v), ops.plus, ops.minus, spacing(p.grid) * sqrt(p.mass), field)
end
function total_energy(p, xs, n, ops, a, rec; time = 0.0, drive = NoDrive())
    h = spacing(p.grid)
    kinetic_terms =
        Q.recorded_map(Float64, length(xs), rec.options, rec; random = false) do i, localrec
            w, x = p.occupations[i], xs[i]
            shifted = Q.apply_mpo(ops.plus, x; tol = a.algebra_tolerance, cap = a.orbital_cap)
            gradient_norm =
                Q.difference_norm(shifted, x; rec = localrec, label = "kinetic_difference_norm")
            w * gradient_norm^2 / (2p.mass * h^2)
        end
    kin = sum(kinetic_terms)
    vh = hartree(n, p, ops, a)
    ext = h * real(Q.inner(n, ops.external; cfg = rec.options))
    eh = 0.5h * real(Q.inner(n, vh; cfg = rec.options))
    exc = 0.0
    if ops.xc_table !== nothing
        en = Q.sampled(n, rec)
        e = Q.cross(
            rec,
            p.grid.R,
            Q.pointwise(value -> Q.xc(ops.xc_table, real(value))[1], en);
            tol = a.xc_energy_tolerance,
            cap = a.field_cap,
            label = "xc_energy_density",
        )
        exc = h * real(Q.inner(Q.constant_tt(p.grid.R), e; cfg = rec.options))
    end
    edrive = 0.0
    if !(drive isa NoDrive)
        v = Q.cross(
            rec,
            p.grid.R,
            j -> drive_value(drive, j * h, time, p.grid.L);
            tol = a.field_tolerance,
            cap = a.field_cap,
            label = "drive_energy",
        )
        edrive = h * real(Q.inner(n, v; cfg = rec.options))
    end
    nn = ops.nuclear
    (;
        total = kin + ext + eh + exc + nn + edrive,
        kinetic = kin,
        external = ext,
        hartree = eh,
        xc = exc,
        nuclear = nn,
        drive = edrive,
    )
end
