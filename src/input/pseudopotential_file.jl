# Read and interpolate radial pseudopotential data for the 3D model.

"""Cubic Hermite interpolation on the supplied radial mesh; no fitting."""
function radial_value(r, values, x)
    x >= 0 && x <= r[end] || throw(BoundsError(r, x))
    x == r[end] && return values[end]
    j = clamp(searchsortedlast(r, x), 1, length(r) - 1)
    h = r[j + 1] - r[j]
    t = (x - r[j]) / h
    slope(i) =
        i == 1 ? 0.0 :
        i == length(r) ? (values[end] - values[end - 1]) / (r[end] - r[end - 1]) :
        (values[i + 1] - values[i - 1]) / (r[i + 1] - r[i - 1])
    (2t^3 - 3t^2 + 1) * values[j] +
    (t^3 - 2t^2 + t) * h * slope(j) +
    (-2t^3 + 3t^2) * values[j + 1] +
    (t^3 - t^2) * h * slope(j + 1)
end
"""Real spherical harmonic for the s and p channels used by this reader."""
function real_harmonic(l, m, displacement, r)
    l == 0 && return inv(sqrt(4pi))
    r == 0 && return 0.0
    sqrt(3 / (4pi)) * displacement[m == -1 ? 2 : m == 0 ? 3 : 1] / r
end

upf_number(s) = parse(Float64, replace(strip(s), 'D' => 'E', 'd' => 'e'))
upf_attributes(text) = Dict(
    m.captures[1] => strip(m.captures[2]) for m in eachmatch(r"([\w_]+)\s*=\s*\"([^\"]*)\"", text)
)
function upf_block(text, name)
    tag = replace(name, "." => "\\.")
    m = match(Regex("<" * tag * "\\b([^>]*)>(.*?)</" * tag * ">", "s"), text)
    m === nothing && error("missing UPF block $name")
    attributes = upf_attributes(m.captures[1])
    numbers = upf_number.(split(m.captures[2]))
    haskey(attributes, "size") &&
        parse(Int, attributes["size"]) != length(numbers) &&
        error("UPF size mismatch for $name")
    all(isfinite, numbers) || error("nonfinite UPF data")
    (; attributes, numbers)
end

"""SALMON/Yabana-Bertsch radial format, with d as the local channel.

Radii are Angstrom, potentials eV, radial functions Angstrom^(-1/2).  The
Kleinman-Bylander beta and D are built exactly from the supplied channels.
"""
function read_silicon_ky(path)
    ang = 1.8897261246257702
    rows = [parse.(Float64, split(line)) for line in eachline(path) if !isempty(strip(line))]
    length(rows[1]) == 4 || error("invalid KY header")
    mr, step, lmax, Z = rows[1]
    isinteger(mr) && mr > 2 && lmax == 2 && Z == 4 && step > 0 ||
        error("four-valence s/p/d KY silicon required")
    n = Int(mr) + 1
    length(rows) == 2 + 2n && length(rows[2]) == 4 || error("KY row count mismatch")
    rows[2][1] == 0 || error("KY core correction is unsupported")
    all(length(row) == 5 for row in rows[3:(2 + n)]) || error("KY potential column count mismatch")
    all(length(row) == 4 for row in rows[(3 + n):end]) || error("KY orbital column count mismatch")
    all(all(isfinite, row) for row in rows) || error("nonfinite KY data")
    radii = [row[1] for row in rows[3:(2 + n)]]
    radii ≈ collect(0:(n - 1)) .* step || error("KY mesh must be uniform")
    [row[1] for row in rows[(3 + n):end]] ≈ radii || error("KY orbital/potential mesh mismatch")
    all(iszero, row[2] for row in rows[3:(2 + n)]) || error("nonzero KY core density unsupported")
    r = radii .* ang
    V = reduce(vcat, [permutedims(row[3:5]) for row in rows[3:(2 + n)]]) ./ 27.211386245988
    u = reduce(vcat, [permutedims(row[2:4]) for row in rows[(3 + n):end]]) ./ sqrt(ang)
    beta = Vector{Float64}[]
    D = zeros(2, 2)
    cuts = Float64[]
    for l = 0:1
        difference = V[:, l + 1] .- V[:, 3]
        integrand = u[:, l + 1] .^ 2 .* difference
        integral = sum(diff(r) .* (integrand[1:(end - 1)] + integrand[2:end]) ./ 2)
        abs(integral) > 1e-12 || error("singular Kleinman-Bylander normalization")
        y = zeros(n)
        y[2:end] = difference[2:end] .* u[2:end, l + 1] ./ r[2:end]
        y[1] = l == 0 ? y[2] : 0.0
        last = findlast(!iszero, y)
        last === nothing && error("empty KY channel")
        last + 2 < n || error("unresolved KY projector tail")
        push!(beta, y)
        push!(cuts, r[last + 2])
        D[l + 1, l + 1] = 1 / integral
    end
    SiliconPseudo(
        r,
        V[:, 3],
        beta,
        [0, 1],
        cuts,
        D,
        4.0,
        abspath(path),
        "SALMON 1.2.0 silicon response tutorial KY asset",
    )
end
