# Periodic soft-Coulomb and 1D LSDA formulas; the XC table returns energy density and its derivative.

using SpecialFunctions, QuadGK

struct XCTable
    log_density::Vector{Float64}
    energy_density::Vector{Float64}
    log_density_derivative::Vector{Float64}
    quadrature_rtol::Float64
end
const HELBIG_CORRELATION_UNPOLARIZED = (18.40, 7.501, 0.10185, 0.012827, 1.511, 0.258, 4.424)
const HELBIG_CORRELATION_POLARIZED = (5.24, 1.568, 0.1286, 0.00320, 0.0538, 1.56e-5, 2.958)

"Helbig et al., unit-softening correlation energy per particle and its r_s derivative."
function helbig_correlation_terms(rho, parameters = HELBIG_CORRELATION_UNPOLARIZED)
    # A, C, D, E, alpha, beta and m are the published fit parameters.
    A, C, D, E, alpha, beta, m = parameters
    rs = 1 / (2rho)
    num = rs + E * rs^2
    den = A + C * rs^2 + D * rs^3
    q = alpha * rs + beta * rs^m
    f = num / den
    fp = ((1 + 2E * rs) * den - num * (2C * rs + 3D * rs^2)) / den^2
    epsilon_c = -0.5 * f * log1p(q)
    derivative_rs = -0.5 * (fp * log1p(q) + f * (alpha + beta * m * rs^(m - 1)) / (1 + q))
    (; rs, epsilon_c, derivative_rs)
end
function correlation(rho)
    terms = helbig_correlation_terms(rho)
    rho * terms.epsilon_c, terms.epsilon_c - terms.rs * terms.derivative_rs
end

"Spin derivatives of n*epsilon_c(r_s,zeta), with quadratic Helbig polarization interpolation."
function helbig_spin_correlation(rho_up, rho_down; density_floor = 1e-14)
    rho = rho_up + rho_down
    rho <= density_floor && return (0.0, 0.0, 0.0)
    unpolarized = helbig_correlation_terms(rho)
    polarized = helbig_correlation_terms(rho, HELBIG_CORRELATION_POLARIZED)
    zeta = (rho_up - rho_down) / rho
    delta_epsilon = polarized.epsilon_c - unpolarized.epsilon_c
    epsilon_c = unpolarized.epsilon_c + zeta^2 * delta_epsilon
    derivative_rs =
        unpolarized.derivative_rs + zeta^2 * (polarized.derivative_rs - unpolarized.derivative_rs)
    potential = epsilon_c - unpolarized.rs * derivative_rs
    derivative_zeta = 2zeta * delta_epsilon
    rho * epsilon_c,
    potential + (1 - zeta) * derivative_zeta,
    potential - (1 + zeta) * derivative_zeta
end

"Central second-derivative coefficients, ordered from the center to positive neighbors."
function finite_difference_coefficients(order::Integer)
    order == 2 && return (-2.0, 1.0)
    order == 8 && return (-205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560)
    throw(ArgumentError("Supported finite-difference orders are 2 and 8"))
end

function kinetic_eigenvalues(n::Integer, h::Real, order::Integer; mass = 1.0)
    coefficients = finite_difference_coefficients(order)
    [
        -(
            coefficients[1] +
            2sum(coefficients[k + 1] * cospi(2k * j / n) for k = 1:(length(coefficients) - 1))
        ) / (2mass * h^2) for j = 0:(n - 1)
    ]
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
    exchange_potential = -i0 / pi
    correlation_energy, correlation_potential = correlation(rho)
    exchange_energy + correlation_energy, exchange_potential + correlation_potential
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
