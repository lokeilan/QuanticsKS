# Periodic soft-Coulomb and 1D LSDA formulas; the XC table returns energy density and its derivative.

using SpecialFunctions, QuadGK

struct XCTable
    logn::Vector{Float64}
    e::Vector{Float64}
    derivative::Vector{Float64}
    quadrature_rtol::Float64
end
function correlation(n)
    rs = 1 / (2n)
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
    eps = -0.5 * f * log1p(q)
    deps = -0.5 * (fp * log1p(q) + f * (alpha + beta * m * rs^(m - 1)) / (1 + q))
    n * eps, eps - rs * deps
end
function xc_exact(n; rtol = 2e-11)
    n = max(n, 1e-12)
    q = pi * n
    i0 = quadgk(t -> besselk(0, t), 0.0, q; rtol = rtol)[1]
    i1 = 1 - q * besselk(1, q)
    ex = -(q * i0 - i1) / pi^2
    vx = -i0 / pi
    ec, vc = correlation(n)
    ex + ec, vx + vc
end
function XCTable(; npoints = 2048, density_min = 1e-8, density_max = 4.0, rtol = 2e-11)
    grid = collect(range(log(density_min), log(density_max); length = npoints))
    es = Float64[]
    ds = Float64[]
    for t in grid
        n = exp(t)
        e, v = xc_exact(n; rtol)
        push!(es, e)
        push!(ds, n * v)
    end
    XCTable(grid, es, ds, rtol)
end
function xc(tab::XCTable, n)
    n = max(real(n), 1e-12)
    t = log(n)
    if t < tab.logn[1] || t > tab.logn[end]
        return xc_exact(n; rtol = tab.quadrature_rtol)
    end
    h = tab.logn[2] - tab.logn[1]
    i = clamp(floor(Int, (t - tab.logn[1]) / h) + 1, 1, length(tab.e) - 1)
    s = (t - tab.logn[i]) / h
    e0 = tab.e[i]
    e1 = tab.e[i + 1]
    d0 = tab.derivative[i]
    d1 = tab.derivative[i + 1]
    e =
        (2s^3 - 3s^2 + 1) * e0 +
        (s^3 - 2s^2 + s) * h * d0 +
        (-2s^3 + 3s^2) * e1 +
        (s^3 - s^2) * h * d1
    dp =
        ((6s^2 - 6s) * e0 + (3s^2 - 4s + 1) * h * d0 + (-6s^2 + 6s) * e1 + (3s^2 - 2s) * h * d1) / h
    e, dp / n
end
const IMAGE_ZETA = [zeta(2m + 1) for m = 1:28]
"Smooth periodic soft Coulomb: symmetric image sum with 2/(mL) constants subtracted."
function soft(x, L)
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
vext(x, L, natoms, spacing) = -sum(soft(x - spacing * A, L) for A = 0:(natoms - 1))
nuclear_energy(L, natoms, spacing) =
    sum(soft(spacing * (A - B), L) for A = 0:(natoms - 1) for B = 0:(A - 1))
