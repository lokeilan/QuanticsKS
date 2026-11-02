# Fit joint QTTs from point samples or k slices; caches live only during the fit.

"Functions receive zero-based spatial and k coordinates. Construction remains joint."
function sample_joint(
    l::JointLayout,
    functions;
    tolerance = 1e-10,
    cap = 64,
    tci = TCIOptions(),
    recorder = nothing,
)
    isfinite(tolerance) && tolerance > 0 && cap > 0 || throw(ArgumentError("invalid accuracy"))
    own = recorder === nothing
    rec = own ? QuanticsKS.recorder(nothing, tci) : recorder
    try
        xs = [
            Q.cross(
                rec,
                sites(l),
                j -> f(decode(l, j)...);
                tol = tolerance,
                cap,
                label = "joint_initial_$b",
            ) for (b, f) in enumerate(functions)
        ]
        JointState(l, xs)
    finally
        own && close(rec)
    end
end

"TCI over all x,k; callback retains at most cache_slices one-k work blocks."
function fit_slices(l, nbands, callback; tol, cap, rec, cache_slices = 8, label = "joint")
    cache = Dict{NTuple{3,Int},Matrix{ComplexF64}}()
    n = shape(l)
    li = LinearIndices(n)
    ks = kcoordinates(l)
    complete = cache_slices >= length(ks)
    blocks = Matrix{ComplexF64}[]
    klinear = LinearIndices(kshape(l))
    if complete
        resize!(blocks, length(ks))
        done = Threads.Atomic{Int}(0)
        println(
            "Building bounded operation workspace: ",
            16prod(n) * nbands * length(ks),
            " bytes; joint QTT state retained",
        )
        flush(stdout)
        Q.timed(rec, "slice_workspace"; item = label) do
            Threads.@threads :static for ik in eachindex(ks)
                block = callback(ks[ik])
                size(block) == (prod(n), nbands) || error("slice dimensions")
                blocks[ik] = block
                count = Threads.atomic_add!(done, 1) + 1
                if count % 1024 == 0 || count == length(ks)
                    println("Workspace ", label, " ", count, "/", length(ks))
                    flush(stdout)
                end
            end
        end
    end
    function slice(k)
        complete && return blocks[klinear[(k .+ 1)...]]
        haskey(cache, k) && return cache[k]
        length(cache) >= cache_slices && delete!(cache, first(keys(cache)))
        value = callback(k)
        size(value) == (prod(n), nbands) || error("slice dimensions")
        cache[k] = value
    end
    orbitals = Q.QTT[]
    for b = 1:nbands
        println("TCI ", label, " orbital ", b, "/", nbands)
        flush(stdout)
        oracle = j -> begin
            x, k = decode(l, j)
            slice(k)[li[(x .+ 1)...], b]
        end
        push!(orbitals, Q.cross(rec, sites(l), oracle; tol, cap, label = "$(label)_$b"))
        rec.dir === nothing || !isfile(joinpath(rec.dir, "STOP")) || throw(InterruptException())
    end
    JointState(l, orbitals)
end
