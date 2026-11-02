# Local and nonlocal pseudopotentials, PZ-LDA, Hartree, and ion energy for silicon.

# Physical KY reader adapted from QuanticsKS silicon_joint.jl (2026-10-31).
# The SCF and joint algebra below are implemented independently in this project.
using SpecialFunctions: erf

# ----------------------------------------------------------------------
# Pseudopotential data
# ----------------------------------------------------------------------
"""Four-valence-electron norm-conserving Si pseudopotential.

`radius` is in bohr, `local_potential` in Ha, `beta_over_r` contains
beta_l(r)/r with the regular origin limit, `dij` is the Kleinman-Bylander
coefficient matrix in Ha.  Only scalar s/p channels are represented.
"""
struct SiliconPseudo
    radius::Vector{Float64}
    local_potential::Vector{Float64}
    beta_over_r::Vector{Vector{Float64}}
    angular_momenta::Vector{Int}
    cutoffs::Vector{Float64}
    dij::Matrix{Float64}
    valence::Float64
    source_path::String
    provenance::String
end

using FFTW, SparseArrays, SpecialFunctions

"Spatial fields plus compact projector image lists; no dense x-by-k orbital storage."
struct SiliconPhysics
    layout::JointLayout
    pseudo::SiliconPseudo
    atoms::Vector{NTuple{3,Float64}}
    local_potential::Vector{Float64}
    g2::Array{Float64,3}
    projector_terms::Vector{Tuple{Int,Int,NTuple{3,Float64},Float64}}
    projector_d::Vector{Float64}
    ion_energy::Float64
end

function silicon_atoms()
    fcc = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
    vcat(fcc, [Tuple(collect(x) .+ 0.25) for x in fcc])
end
signedmode(i, n) = i <= n ÷ 2 ? i - 1 : i - 1 - n

"Point-ion Ewald energy including the neutralizing G=0 convention."
function ewald_energy(atoms, a; alpha = 5 / a, shell = 12, Z = 4.0)
    realpart = 0.0
    reciprocal = 0.0
    for i in eachindex(atoms),
        j in eachindex(atoms),
        x = (-shell):shell,
        y = (-shell):shell,
        z = (-shell):shell

        i == j && x == y == z == 0 && continue
        d = a .* (collect(atoms[i]) .- collect(atoms[j]) .+ [x, y, z])
        r = norm(d)
        realpart += 0.5Z^2 * erfc(alpha * r) / r
    end
    for x = (-shell):shell, y = (-shell):shell, z = (-shell):shell
        x == y == z == 0 && continue
        g2 = (2pi / a)^2 * (x * x + y * y + z * z)
        structure = sum(Z * cis(-2pi * (x * r[1] + y * r[2] + z * r[3])) for r in atoms)
        reciprocal += 2pi / a^3 * abs2(structure) * exp(-g2 / (4alpha^2)) / g2
    end
    realpart + reciprocal - alpha / sqrt(pi) * length(atoms) * Z^2 -
    pi * (length(atoms) * Z)^2 / (2alpha^2 * a^3)
end

function build_physics(l::JointLayout, pseudo_path; ionic_width = 0.8)
    p = read_silicon_ky(pseudo_path)
    atoms = silicon_atoms()
    n = shape(l)
    N = prod(n)
    a = l.a_bohr
    gs = [ntuple(ax -> 2pi / a * signedmode(I[ax], n[ax]), 3) for I in CartesianIndices(n)]
    g2 = map(g -> sum(abs2, g), gs)
    nodes = (-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526)
    weights = (0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538)
    rs = Float64[]
    ws = Float64[]
    for i = 1:(length(p.radius) - 1), (t, w) in zip(nodes, weights)
        h = p.radius[i + 1] - p.radius[i]
        r = (p.radius[i + 1] + p.radius[i] + h * t) / 2
        short = local_potential_value(p, r) + p.valence * erf(r / ionic_width) / r
        push!(rs, r)
        push!(ws, 4pi * r^2 * short * h * w / 2)
    end
    cache = Dict{Int,Float64}()
    coeff = zeros(ComplexF64, n)
    for I in CartesianIndices(n)
        g = gs[I]
        s = g2[I]
        key = sum(signedmode(I[ax], n[ax])^2 for ax = 1:3)
        radial = get!(cache, key) do
            sum(ws .* sinc.(sqrt(s) .* rs ./ pi)) -
            (s == 0 ? 0.0 : 4pi * p.valence * exp(-ionic_width^2 * s / 4) / s)
        end
        coeff[I] = radial * sum(cis(-a * sum(g[ax] * r[ax] for ax = 1:3)) for r in atoms) / a^3
    end
    # Explicit collocation Nyquist symmetrization, matching native reference.
    sym = similar(coeff)
    for I in CartesianIndices(n)
        partner = ntuple(ax -> mod(-(I[ax] - 1), n[ax]) + 1, 3)
        sym[I] = 0.5 * (coeff[I] + conj(coeff[partner...]))
    end
    potential = vec(real.(ifft(sym)) .* N)
    terms = Tuple{Int,Int,NTuple{3,Float64},Float64}[]
    D = Float64[]
    col = 0
    for atom in atoms, channel = 1:2, mm = (-p.angular_momenta[channel]):p.angular_momenta[channel]
        col += 1
        push!(D, p.dij[channel, channel])
        cut = p.cutoffs[channel]
        for I in CartesianIndices(n)
            fraction = ntuple(ax -> (I[ax] - 1) / n[ax] - atom[ax], 3)
            ranges = ntuple(
                ax -> ceil(Int, fraction[ax] - cut / a):floor(Int, fraction[ax] + cut / a),
                3,
            )
            for image in Iterators.product(ranges...)
                d = ntuple(ax -> a * (fraction[ax] - image[ax]), 3)
                r = sqrt(sum(abs2, d))
                r >= cut && continue
                value =
                    radial_value(p.radius, p.beta_over_r[channel], r) *
                    real_harmonic(p.angular_momenta[channel], mm, d, r)
                iszero(value) || push!(terms, (LinearIndices(n)[I], col, d, value * sqrt(dv(l))))
            end
        end
    end
    SiliconPhysics(l, p, atoms, potential, g2, terms, D, ewald_energy(atoms, a))
end

"Unpolarized PZ81 LDA: epsilon_xc per electron and derivative d(n epsilon)/dn."
function pz_lda(n)
    n >= 0 || error("negative physical density")
    n == 0 && return (0.0, 0.0)
    rs = (3 / (4pi * n))^(1 / 3)
    if rs >= 1
        den = 1 + 1.0529sqrt(rs) + 0.3334rs
        ec = -0.1423 / den
        dec = 0.1423 * (0.52645 / sqrt(rs) + 0.3334) / den^2
    else
        ec = 0.0311log(rs) - 0.048 + 0.002rs * log(rs) - 0.0116rs
        dec = 0.0311 / rs + 0.002 * (log(rs) + 1) - 0.0116
    end
    ex = -0.75 * (3 / pi)^(1 / 3) * n^(1 / 3)
    (ex + ec, 4ex / 3 + ec - rs * dec / 3)
end
function hartree_potential(p::SiliconPhysics, rho)
    n = shape(p.layout)
    f = fft(reshape(rho, n))
    f[1] = 0
    for I in CartesianIndices(n)
        p.g2[I] == 0 || (f[I] *= 4pi / p.g2[I])
    end
    vec(real.(ifft(f)))
end
function physical_fields(p, rho)
    vh = hartree_potential(p, rho)
    xc = pz_lda.(rho)
    (;
        potential = p.local_potential .+ vh .+ last.(xc),
        hartree = vh,
        xc_energy = dv(p.layout) * sum(rho .* first.(xc)),
    )
end
function initial_density(p; sigma = 0.6)
    l = p.layout
    n = shape(l)
    rho = zeros(n)
    for I in CartesianIndices(n), atom in p.atoms
        d = ntuple(ax -> begin
            f = (I[ax] - 1) / n[ax] - atom[ax]
            l.a_bohr * (f - round(f))
        end, 3)
        rho[I] += exp(-sum(abs2, d) / (2sigma^2))
    end
    rho .*= 32 / (dv(l) * sum(rho))
    vec(rho)
end
