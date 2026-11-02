# Periodic soft-Coulomb potentials, Hartree kernel, and XC selected by physical_model.

minimum_image(d, L) = mod(d + L / 2, L) - L / 2
function HartreeWorkspace(n, L)
    dx = L / n
    w = FFTWorkspace(n)
    for j = 0:(n - 1)
        w.buffer[j + 1] = inv(sqrt(minimum_image(j * dx, L)^2 + 1))
    end
    w.forward * w.buffer
    return HartreeWorkspace(w, copy(w.buffer), dx)
end
function HartreeWorkspace(n, L, c::Config)
    c.physical_model == "minimum_image" && return HartreeWorkspace(n, L)
    dx = L / n
    w = FFTWorkspace(n)
    Threads.@threads :static for j = 0:(n - 1)
        w.buffer[j + 1] = QTTReferencePhysics.soft(j * dx, L)
    end
    w.forward * w.buffer
    return HartreeWorkspace(w, copy(w.buffer), dx)
end

prepare_xc(c::Config) =
    c.physical_model == "minimum_image" ? nothing :
    QTTReferencePhysics.XCTable(
        npoints = c.xc_table_points,
        density_min = c.xc_density_min,
        density_max = c.xc_density_max,
        rtol = c.exchange_quadrature_rtol,
    )

xc!(v, rho, dx, c::Config, ::Nothing) = xc!(v, rho, dx, c.xc_rtol)
function xc!(v, rho, dx, c::Config, tab::QTTReferencePhysics.XCTable)
    chunks = Threads.nthreads()
    ex = zeros(chunks)
    ec = zeros(chunks)
    Threads.@threads :static for chunk = 1:chunks
        first = 1 + (chunk - 1) * length(rho) ÷ chunks
        last = chunk * length(rho) ÷ chunks
        sx = 0.0
        sc = 0.0
        @inbounds for i = first:last
            energy, potential = QTTReferencePhysics.xc(tab, rho[i])
            correlation_energy, _ = QTTReferencePhysics.correlation(max(rho[i], 1e-12))
            v[i] = potential
            sx += energy - correlation_energy
            sc += correlation_energy
        end
        ex[chunk] = sx
        ec[chunk] = sc
    end
    return (; exchange = dx * sum(ex), correlation = dx * sum(ec))
end

function model_description(c::Config)
    qtt = c.physical_model == "qtt_periodic"
    return Dict(
        "physical_model" => c.physical_model,
        "kernel" =>
            qtt ?
            "QTT smooth periodic soft Coulomb; symmetric image sum with 2/(mL) constants subtracted; 28 even Legendre terms" :
            "minimum-image soft Coulomb",
        "coordinates" =>
            qtt ? "x_j=j*L/N; nuclei A*spacing, A=0:natoms-1" :
            "centered cell; half-spacing centered nuclei",
        "xc" =>
            qtt ?
            "QTT 2048-point log-density cubic Hermite table, matching energy derivative and density floor" :
            "direct exchange quadrature and Helbig correlation",
        "zero_frequency" => "retained; no additional background subtraction",
        "ion_ion" => "same kernel; distinct in-cell nuclear pairs only, matching QTT",
        "cold_start" => true,
        "spatial_symmetry_reduction" => false,
    )
end
"""Unpolarized soft-Coulomb exchange, direct quadrature at each density; a=1."""
function exchange_direct(n::Float64, rtol)
    n == 0 && return (epsilon = 0.0, potential = 0.0)
    n > 0 && isfinite(n) || error("Invalid density")
    # u=t^2 makes the integrable logarithmic endpoint finite. No lookup table.
    ex, exerr =
        quadgk(t -> t == 0 ? 0.0 : 2t * (1 - t * t) * besselk(0, pi * n * t * t), 0.0, 1.0; rtol)
    vx, vxerr = quadgk(t -> t == 0 ? 0.0 : 2t * besselk(0, pi * n * t * t), 0.0, 1.0; rtol)
    exerr <= rtol * abs(ex) && vxerr <= rtol * abs(vx) || error("Exchange quadrature failed")
    return (epsilon = -n * ex, potential = -n * vx)
end

function exchange(n::Float64, rtol)
    # Integrate the logarithmic term analytically. The remainder is the same
    # Bessel integral, evaluated directly at every grid point without a table.
    (n <= 0.01 || n >= 4.0) && return exchange_direct(n, rtol)
    a = pi * n
    gamma = Float64(MathConstants.eulergamma)
    g = log(a / 2) + gamma
    value, err = quadgk(0.0, 1.0; rtol = rtol / 32) do t
        t == 0 && return 0.0 + 0.0im
        z = a * t * t
        remainder = besselk(0, z) + log(z / 2) + gamma
        2t * remainder * complex(1 - t * t, 1.0)
    end
    ix = 0.75 - 0.5g + real(value)
    iv = 1 - g + imag(value)
    err <= rtol * min(abs(ix), abs(iv)) || return exchange_direct(n, rtol)
    return (epsilon = -n * ix, potential = -n * iv)
end

"""Helbig et al. PRA 83 032503 (2011), Eq. 4/Table I, unpolarized a=1."""
function correlation(n::Float64)
    n == 0 && return (epsilon = 0.0, potential = 0.0)
    n > 0 && isfinite(n) || error("Invalid density")
    rs = 1 / (2n)
    A, C, D, E, alpha, beta, m = 18.40, 7.501, 0.10185, 0.012827, 1.511, 0.258, 4.424
    p = rs + E * rs^2
    dp = 1 + 2E * rs
    q = A + C * rs^2 + D * rs^3
    dq = 2C * rs + 3D * rs^2
    z = alpha * rs + beta * rs^m
    ell = log1p(z)
    dell = (alpha + beta * m * rs^(m - 1)) / (1 + z)
    eps = -0.5 * p / q * ell
    deps = -0.5 * ((dp * q - p * dq) / q^2 * ell + p / q * dell)
    return (epsilon = eps, potential = eps - rs * deps)
end
function xc!(v, rho, dx, rtol)
    chunks = Threads.nthreads()
    ex = zeros(chunks)
    ec = zeros(chunks)
    Threads.@threads :static for chunk = 1:chunks
        first = 1 + (chunk - 1) * length(rho) ÷ chunks
        last = chunk * length(rho) ÷ chunks
        local_ex = 0.0
        local_ec = 0.0
        @inbounds for i = first:last
            x = exchange(rho[i], rtol)
            c = correlation(rho[i])
            v[i] = x.potential + c.potential
            local_ex += rho[i] * x.epsilon
            local_ec += rho[i] * c.epsilon
        end
        ex[chunk] = local_ex
        ec[chunk] = local_ec
    end
    total_ex = sum(ex)
    total_ec = sum(ec)
    all(isfinite, v) && isfinite(total_ex + total_ec) || error("Nonfinite XC")
    return (exchange = dx * total_ex, correlation = dx * total_ec)
end

function geometry(c)
    if c.physical_model == "qtt_periodic"
        n = 2^c.R
        L = c.natoms * c.spacing
        dx = L / n
        x = collect(0:(n - 1)) .* dx
        nuclei = collect(0:(c.natoms - 1)) .* c.spacing
        v = zeros(n)
        rho = zeros(n)
        Threads.@threads :static for i in eachindex(x)
            v[i] = -sum(QTTReferencePhysics.soft(x[i] - a, L) for a in nuclei)
            rho[i] = sum(exp(-minimum_image(x[i] - a, L)^2) for a in nuclei)
        end
        rho .*= c.natoms / (dx * sum(rho))
        eii = sum(
            (
                QTTReferencePhysics.soft(nuclei[a] - nuclei[b], L) for a in eachindex(nuclei)
                for b = 1:(a - 1)
            );
            init = 0.0,
        )
        return (; n, L, dx, x, nuclei, v, rho, eii)
    end
    n = 2^c.R
    L = c.natoms * c.spacing
    dx = L / n
    x = [-L / 2 + (i - 1) * dx for i = 1:n]
    nuclei = [(j - (c.natoms + 1) / 2) * c.spacing for j = 1:(c.natoms)]
    v = [-sum(inv(sqrt(minimum_image(xi - X, L)^2 + 1)) for X in nuclei) for xi in x]
    rho = [sum(exp(-minimum_image(xi - X, L)^2) for X in nuclei) for xi in x]
    rho .*= c.natoms / (dx * sum(rho))  # Only the initial trial density is normalized.
    eii = sum(
        inv(sqrt(minimum_image(nuclei[i] - nuclei[j], L)^2 + 1)) for i = 1:(c.natoms) for
        j = (i + 1):(c.natoms)
    )
    return (; n, L, dx, x, nuclei, v, rho, eii)
end
