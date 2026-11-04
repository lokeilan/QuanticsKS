# Project and rebuild tangent vectors in one orthogonal frame with shared core offsets.

struct Frame
    left::Vector{Array{ComplexF64,3}}
    right::Vector{Array{ComplexF64,3}}
    shapes::Vector{Tuple{Int,Int,Int}}
    offsets::Vector{Int}
end
function canonical_frame(x::QTT; cfg = (;))
    sweeps = independent_map(Any, 2, cfg) do side
        train = side == 1 ? deepcopy(tensortrain(x)) : reverse(tensortrain(x))
        with_svd_context(;
            operation = side == 1 ? "canonical_right" : "canonical_left",
            original_cores = x.c,
        ) do
            guarded_compress!(train; tolerance = 0.0)
        end
        train
    end
    right, left = sweeps
    u = reverse(left).sitetensors
    v = right.sitetensors
    d = length(x)
    sh = [(size(u[k], 1), 2, size(v[k], 3)) for k = 1:d]
    off = cumsum(vcat(0, [prod(s) for s in sh]))
    Frame(u, v, sh, off)
end
function pack_delta(f::Frame, gs)
    reduce(vcat, vec.(gs))
end
function unpack_delta(f::Frame, z)
    [reshape(view(z, (f.offsets[k] + 1):f.offsets[k + 1]), f.shapes[k]) for k = 1:length(f.shapes)]
end
# Remove the component parallel to the left basis to avoid redundant tangent coordinates.
function gauge!(g, u)
    m = reshape(g, :, size(g, 3))
    q = reshape(u, :, size(u, 3))
    m .-= q * (q' * m)
    g
end
"Project an MPO action directly into tangent coordinates, without constructing W*x."
function project(f::Frame, x::QTT, w = nothing)
    d = length(x)
    if w === nothing
        w = [reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2, 1) for k = 1:d]
    end
    right = Vector{Array{ComplexF64,3}}(undef, d + 1)
    right[d + 1] = ones(ComplexF64, 1, 1, 1)
    for k = d:-1:1
        v = f.right[k]
        a = w[k]
        z = x.c[k]
        t = right[k + 1]
        @tensor t1[l, s, I, J] := z[l, s, L] * t[I, J, L]
        @tensor t2[j, s, l, I] := a[j, s, t, J] * t1[l, t, I, J]
        @tensor e[i, j, l] := conj(v[i, s, I]) * t2[j, s, l, I]
        right[k] = e
    end
    left = ones(ComplexF64, 1, 1, 1)
    ds = Array{ComplexF64,3}[]
    for k = 1:d
        a = w[k]
        z = x.c[k]
        t = right[k + 1]
        u = f.left[k]
        @tensor t1[i, j, s, L] := left[i, j, l] * z[l, s, L]
        @tensor t2[i, s, J, L] := t1[i, j, t, L] * a[j, s, t, J]
        @tensor g[i, s, I] := t2[i, s, J, L] * t[I, J, L]
        k < d && gauge!(g, u)
        push!(ds, g)
        if k < d
            @tensor e[I, J, L] := conj(u[i, s, I]) * t2[i, s, J, L]
            left = e
        end
    end
    pack_delta(f, ds)
end
function tangent_tt(f::Frame, z)
    ds = unpack_delta(f, z)
    c = Array{ComplexF64,3}[]
    d = length(ds)
    for k = 1:d
        u = f.left[k]
        v = f.right[k]
        a = ds[k]
        if k == 1
            g = cat(a, u; dims = 3)
        elseif k == d
            g = cat(v, a; dims = 1)
        else
            g = zeros(ComplexF64, size(v, 1) + size(u, 1), 2, size(v, 3) + size(u, 3))
            g[1:size(v, 1), :, 1:size(v, 3)] = v
            g[(size(v, 1) + 1):end, :, 1:size(v, 3)] = a
            g[(size(v, 1) + 1):end, :, (size(v, 3) + 1):end] = u
        end
        push!(c, g)
    end
    QTT(c)
end
