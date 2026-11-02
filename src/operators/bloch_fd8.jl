# Bloch FD8 and nonlocal actions on columns normalized as sqrt(dV)*u.

# Eighth-order second derivative, offsets -4:4. Shared wrapped grid for every k.
const FD8 = [-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560]

"Bloch-phase FD8 action at one joint sample; evaluator accepts a packed label."
function kinetic_at(l::JointLayout, evaluator, x, k)
    encode(l, x, k)
    n = shape(l)
    kfrac = kpoint(l, k)
    result = 0.0im
    for axis = 1:3
        h = l.a_bohr / n[axis]
        for offset = -4:4
            neighbor = ntuple(a -> a == axis ? mod(x[a] + offset, n[a]) : x[a], 3)
            phase = cis(2pi * kfrac[axis] * offset / n[axis])
            result -= FD8[offset + 5] * phase * evaluator(encode(l, neighbor, k)) / (2h^2)
        end
    end
    result
end
function kinetic_symbol(p::SiliconPhysics, k)
    l = p.layout
    n = shape(l)
    kf = kpoint(l, k)
    t = zeros(n)
    for I in CartesianIndices(n), ax = 1:3
        phase = 2pi * (signedmode(I[ax], n[ax]) + kf[ax]) / n[ax]
        t[I] -= sum(FD8[o + 5] * cos(o * phase) for o = -4:4) / (2 * (l.a_bohr / n[ax])^2)
    end
    t
end
function projector_matrix(p::SiliconPhysics, k)
    q = (2pi / p.layout.a_bohr) .* kpoint(p.layout, k)
    rows = Int[]
    cols = Int[]
    values = ComplexF64[]
    for (row, col, d, v) in p.projector_terms
        push!(rows, row)
        push!(cols, col)
        push!(values, v * cis(-sum(q[ax] * d[ax] for ax = 1:3)))
    end
    sparse(rows, cols, values, prod(shape(p.layout)), length(p.projector_d))
end

"Hamiltonian action on sqrt(dV) u columns; bounded one-k scratch."
function haction(p, k, X, potential; parts = false)
    n = shape(p.layout)
    t = kinetic_symbol(p, k)
    B = projector_matrix(p, k)
    TX = similar(X)
    for b in axes(X, 2)
        TX[:, b] = vec(ifft(t .* fft(reshape(view(X, :, b), n))))
    end
    NL = B * (p.projector_d .* (B' * X))
    HX = TX + NL + potential .* X
    parts ? (; HX, TX, NL) : HX
end
function green_action(p, k, X, mu)
    n = shape(p.layout)
    den = kinetic_symbol(p, k) .+ mu
    out = similar(X)
    for b in axes(X, 2)
        out[:, b] = vec(ifft(fft(reshape(view(X, :, b), n)) ./ den))
    end
    out
end

"Independent dense coordinate-space FD matrix plus explicit nonlocal projectors."
function dense_hamiltonian(p, k, potential)
    l = p.layout
    n = shape(l)
    N = prod(n)
    T = zeros(ComplexF64, N, N)
    kf = kpoint(l, k)
    for I in CartesianIndices(n), ax = 1:3, o = -4:4
        neighbor = ntuple(j -> j == ax ? mod(I[j] - 1 + o, n[j]) + 1 : I[j], 3)
        T[LinearIndices(n)[I], LinearIndices(n)[neighbor...]] -=
            FD8[o + 5] * cis(2pi * kf[ax] * o / n[ax]) / (2 * (l.a_bohr / n[ax])^2)
    end
    B = projector_matrix(p, k)
    Hermitian(T + Matrix(B * Diagonal(p.projector_d) * B') + Diagonal(potential))
end
local_potential_value(p::SiliconPseudo, r) =
    r > p.radius[end] ? -p.valence / r : radial_value(p.radius, p.local_potential, r)
