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

struct Operators{W,T}
    external::Q.QTT
    interaction::W
    xc_table::T
    plus::Vector{Array{ComplexF64,4}}
    minus::Vector{Array{ComplexF64,4}}
end
function build_operators(p, a, rec, physics::PhysicsNumerics)
    validate(physics)
    g = p.grid
    h = spacing(g)
    N = length(g)
    pivots =
        p.model isa SoftCoulombLSDA1D ? round.(Int, mod.(p.model.positions, g.L) ./ h) .% N : Int[]
    external =
        p.model isa LocalPotential1D && p.model.potential isa ConstantPotential ?
        Q.constant_tt(g.R, p.model.potential.value) :
        Q.cross(
            rec,
            g.R,
            j -> external_value(p.model, j * h, g.L);
            tol = a.external_tolerance,
            cap = a.field_cap,
            label = "external",
            pivots,
        )
    w =
        p.model isa SoftCoulombLSDA1D ?
        Q.circulant(
            Q.cross(
                rec,
                g.R,
                j -> kernel_value(p.model, j * h, g.L);
                tol = a.kernel_tolerance,
                cap = a.field_cap,
                label = "interaction",
            ),
        ) : nothing
    tab =
        has_xc(p.model) ?
        Q.XCTable(
            npoints = physics.xc_table_points,
            density_min = physics.xc_density_min,
            density_max = physics.xc_density_max,
            rtol = physics.exchange_quadrature_rtol,
        ) : nothing
    Operators(
        external,
        w,
        tab,
        Q.circulant(Q.delta_tt(g.R, 1)),
        Q.circulant(Q.delta_tt(g.R, N - 1)),
    )
end
# Cores store sqrt(h)*psi, so divide by h to recover physical density.
function density(p, xs, a, rec; renormalize = false, label = "density")
    h = spacing(p.grid)
    o = Q.block_oracle(xs, rec)
    n = Q.cross(
        rec,
        p.grid.R,
        j -> sum(p.occupations .* abs2.(o(j))) / h;
        tol = a.density_tolerance,
        cap = a.field_cap,
        label,
    )
    if renormalize
        charge = h * real(Q.inner(Q.constant_tt(p.grid.R), n))
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
function effective_hamiltonian(p, n, ops, a, rec; time = 0.0, drive = NoDrive())
    if p.model isa LocalPotential1D && drive isa NoDrive
        v = ops.external
        return Q.Hamiltonian(v, Q.diagonal(v), ops.plus, ops.minus, spacing(p.grid) * sqrt(p.mass))
    end
    vh = hartree(n, p, ops, a)
    en = Q.evaluate(n, rec)
    eh = Q.evaluate(vh, rec)
    ev = Q.evaluate(ops.external, rec)
    h = spacing(p.grid)
    v = Q.cross(
        rec,
        p.grid.R,
        j ->
            real(ev(j) + eh(j)) +
            (ops.xc_table === nothing ? 0.0 : Q.xc(ops.xc_table, real(en(j)))[2]) +
            drive_value(drive, j * h, time, p.grid.L);
        tol = a.field_tolerance,
        cap = a.field_cap,
        label = "effective_potential",
    )
    Q.Hamiltonian(v, Q.diagonal(v), ops.plus, ops.minus, h * sqrt(p.mass))
end
function total_energy(p, xs, n, ops, a, rec; time = 0.0, drive = NoDrive())
    h = spacing(p.grid)
    kin = 0.0
    for (w, x) in zip(p.occupations, xs)
        shifted = Q.apply_mpo(ops.plus, x; tol = a.algebra_tolerance, cap = a.orbital_cap)
        gradient_norm = Q.difference_norm(shifted, x; rec, label = "kinetic_difference_norm")
        kin += w * gradient_norm^2 / (2p.mass * h^2)
    end
    vh = hartree(n, p, ops, a)
    ext = h * real(Q.inner(n, ops.external))
    eh = 0.5h * real(Q.inner(n, vh))
    exc = 0.0
    if ops.xc_table !== nothing
        en = Q.evaluate(n, rec)
        e = Q.cross(
            rec,
            p.grid.R,
            j -> Q.xc(ops.xc_table, real(en(j)))[1];
            tol = a.xc_energy_tolerance,
            cap = a.field_cap,
            label = "xc_energy_density",
        )
        exc = h * real(Q.inner(Q.constant_tt(p.grid.R), e))
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
        edrive = h * real(Q.inner(n, v))
    end
    nn = nuclear_energy(p.model, p.grid.L)
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
