# Periodic soft-Coulomb and 1D LSDA formulas; the XC table returns energy density and its derivative.

using SpecialFunctions, QuadGK

struct XCTable
    log_density::Vector{Float64}
    energy_density::Vector{Float64}
    log_density_derivative::Vector{Float64}
    quadrature_rtol::Float64
end
function correlation(rho)
    # Helbig et al., Eq. (5) and Table 1: unpolarized 1D gas, unit softening.
    # A, C, D, E, alpha, beta, and m retain the parametrization's symbols.
    rs = 1 / (2rho)
    A = 18.40
    C = 7.501
    D = 0.10185
    E = 0.012827
    alpha = 1.511
    beta = 0.258
    m = 4.424
    num = rs + E * rs^2
    den = A + C * rs^2 + D * rs^3
    q = alpha * rs + beta * rs^m
    f = num / den
    fp = ((1 + 2E * rs) * den - num * (2C * rs + 3D * rs^2)) / den^2
    epsilon_c = -0.5 * f * log1p(q)
    derivative_rs = -0.5 * (fp * log1p(q) + f * (alpha + beta * m * rs^(m - 1)) / (1 + q))
    rho * epsilon_c, epsilon_c - rs * derivative_rs
end
function xc_exact(rho; rtol = 2e-11)
    rho = max(rho, 1e-12)
    q = pi * rho
    i0 = quadgk(t -> besselk(0, t), 0.0, q; rtol = rtol)[1]
    # Avoid cancellation in 1-q*K1(q); table derivatives amplify that noise at low density.
    i1 =
        q < 0.1 ? q^2 * quadgk(t -> t * besselk(0, q * t), 0.0, 1.0; rtol)[1] :
        1 - q * besselk(1, q)
    exchange_energy = -(q * i0 - i1) / pi^2
    Vexchange = -i0 / pi
    correlation_energy, Vcorrelation = correlation(rho)
    exchange_energy + correlation_energy, Vexchange + Vcorrelation
end
function XCTable(;
    npoints = 2048,
    density_min = 1e-8,
    density_max = 4.0,
    rtol = 2e-11,
    parallel_tasks = 1,
)
    log_density = collect(range(log(density_min), log(density_max); length = npoints))
    energy_density = Vector{Float64}(undef, npoints)
    log_density_derivative = Vector{Float64}(undef, npoints)
    foreach_independent(npoints, (; parallel_tasks)) do i
        rho = exp(log_density[i])
        exc, Vxc = xc_exact(rho; rtol)
        energy_density[i] = exc
        log_density_derivative[i] = rho * Vxc
    end
    XCTable(log_density, energy_density, log_density_derivative, rtol)
end
function xc(table::XCTable, rho)
    rho = max(real(rho), 1e-12)
    log_rho = log(rho)
    if log_rho < table.log_density[1] || log_rho > table.log_density[end]
        return xc_exact(rho; rtol = table.quadrature_rtol)
    end
    h = table.log_density[2] - table.log_density[1]
    i = clamp(
        floor(Int, (log_rho - table.log_density[1]) / h) + 1,
        1,
        length(table.energy_density) - 1,
    )
    s = (log_rho - table.log_density[i]) / h
    e0, e1 = table.energy_density[i], table.energy_density[i + 1]
    d0, d1 = table.log_density_derivative[i], table.log_density_derivative[i + 1]
    energy_density =
        (2s^3 - 3s^2 + 1) * e0 +
        (s^3 - 2s^2 + s) * h * d0 +
        (-2s^3 + 3s^2) * e1 +
        (s^3 - s^2) * h * d1
    dp =
        ((6s^2 - 6s) * e0 + (3s^2 - 4s + 1) * h * d0 + (-6s^2 + 6s) * e1 + (3s^2 - 2s) * h * d1) / h
    energy_density, dp / rho
end
const IMAGE_ZETA = [zeta(2m + 1) for m = 1:28]
"Smooth periodic soft Coulomb: symmetric image sum with 2/(mL) constants subtracted."
function periodic_soft_coulomb(x, L)
    x = mod(x + L / 2, L) - L / 2
    r = sqrt(1 + x * x)
    t = x / r
    q = r / L
    p0 = 1.0
    p1 = t
    power = 1.0
    tail = 0.0
    for k = 2:56
        p = ((2k - 1) * t * p1 - (k - 1) * p0) / k
        p0, p1 = p1, p
        if iseven(k)
            power *= q * q
            tail += p * power * IMAGE_ZETA[k ÷ 2]
        end
    end
    inv(r) + 2tail / L
end
