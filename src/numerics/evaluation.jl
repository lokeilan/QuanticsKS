# Evaluate integer-labelled samples; the first core holds the least-significant bit.

mutable struct Evaluator
    x::QTT
    a::Vector{ComplexF64}
    b::Vector{ComplexF64}
    left::Matrix{ComplexF64}
    right::Matrix{ComplexF64}
    prefix::Int
    suffix::Int
end
function Evaluator(x::QTT; cachebits = 10)
    0 <= cachebits <= 16 || throw(ArgumentError("Environment cache bits must be 0:16"))
    # cachebits is the number of cached binary sites at each end, not a
    # dictionary-entry budget. Retain only the endpoint environments. R16
    # with cachebits=8 needs two sets of 256 vectors and one dot product/query.
    # Larger grids contract the uncached middle cores using reusable buffers.
    p = min(cachebits, length(x) ÷ 2)
    s = min(cachebits, length(x) - p)
    l = ones(ComplexF64, 1, 1)
    for k = 1:p
        g = x.c[k]
        next = zeros(ComplexF64, size(g, 3), 2size(l, 2))
        for bit = 1:2
            next[:, ((bit - 1) * size(l, 2) + 1):(bit * size(l, 2))] = transpose(g[:, bit, :]) * l
        end
        l = next
    end
    r = ones(ComplexF64, 1, 1)
    for k = length(x):-1:(length(x) - s + 1)
        g = x.c[k]
        next = zeros(ComplexF64, size(g, 1), 2size(r, 2))
        for bit = 1:2
            next[:, bit:2:end] = g[:, bit, :] * r
        end
        r = next
    end
    Evaluator(x, zeros(ComplexF64, max(rank(x), 1)), zeros(ComplexF64, max(rank(x), 1)), l, r, p, s)
end
function (e::Evaluator)(j::Int)
    0 <= j < (1 << length(e.x)) || throw(BoundsError(e.x, j))
    li = (j & ((1 << e.prefix) - 1)) + 1
    ri = (j >> (length(e.x) - e.suffix)) + 1
    if e.prefix + e.suffix == length(e.x)
        zr = 0.0
        zi = 0.0
        @inbounds @simd for a = 1:size(e.left, 1)
            l = e.left[a, li]
            r = e.right[a, ri]
            zr += real(l) * real(r) - imag(l) * imag(r)
            zi += real(l) * imag(r) + imag(l) * real(r)
        end
        return ComplexF64(zr, zi)
    end
    @inbounds for a = 1:size(e.left, 1)
        e.a[a] = e.left[a, li]
    end
    for k = (e.prefix + 1):(length(e.x) - e.suffix)
        g = e.x.c[k]
        s = ((j >> (k - 1)) & 1) + 1
        @inbounds for b = 1:size(g, 3)
            z = zero(ComplexF64)
            for a = 1:size(g, 1)
                z += e.a[a] * g[a, s, b]
            end
            e.b[b] = z
        end
        e.a, e.b = e.b, e.a
    end
    z = zero(ComplexF64)
    @inbounds for a = 1:size(e.right, 1)
        z += e.a[a] * e.right[a, ri]
    end
    z
end
bits(j, R) = [((j >> (k - 1)) & 1) + 1 for k = 1:R]
indexof(v) = sum(((v[k] - 1) << (k - 1) for k = 1:length(v)); init = 0)
