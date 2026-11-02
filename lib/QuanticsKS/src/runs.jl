# Validate options and manage run logs; collect source snapshots from this src directory.

option_dict(x) = Dict(string(k) => getfield(x, k) for k in fieldnames(typeof(x)))
option_tuple(x) = (; (k => getfield(x, k) for k in fieldnames(typeof(x)))...)
function validate(t::TCIOptions)
    min(
        t.max_iterations,
        t.check_history,
        t.scalar_cache_capacity,
        t.orbital_cache_capacity,
        t.zero_check_limit,
    ) > 0 || throw(ArgumentError("Invalid TCI counts"))
    min(t.global_pivot_search, t.max_global_pivots, t.initial_random_pivots) >= 0 ||
        throw(ArgumentError("Negative pivot count"))
    0 <= t.environment_cache_bits <= 16 ||
        throw(ArgumentError("Environment cache bits must be 0:16"))
    isfinite(t.post_rounding_factor) && 0 < t.post_rounding_factor <= 1 ||
        throw(ArgumentError("Invalid post-rounding factor"))
    t
end
recorder(dir, t::TCIOptions) = Q.Recorder(dir; options = option_tuple(validate(t)))
function validate(r::RefinementOptions)
    r.degree in (1, 3, 5, 7, 9) &&
        1 <= r.max_added_bits <= 12 &&
        isfinite(r.tolerance) &&
        r.tolerance > 0 || throw(ArgumentError("Invalid refinement options"))
    r
end
function validate(p::PhysicsNumerics)
    p.xc_table_points >= 4 &&
        0 < p.xc_density_min < p.xc_density_max &&
        isfinite(p.xc_density_max) &&
        0 < p.exchange_quadrature_rtol < 1 || throw(ArgumentError("Invalid XC numerics"))
    p
end
function validate(g::InitialGuess)
    all(isfinite, (g.envelope_amplitude, g.slow_cosine_amplitude, g.slow_sine_amplitude)) ||
        throw(ArgumentError("Initial amplitudes must be finite"))
    g.envelope_period === nothing ||
        (isfinite(g.envelope_period) && g.envelope_period > 0) ||
        throw(ArgumentError("Invalid envelope period"))
    g
end
function validate(s::SCFOptions)
    s.criterion in (:energy, :strict) ||
        throw(ArgumentError("SCF criterion must be :energy or :strict"))
    min(s.max_iterations, s.inner_steps, s.coefficient_sweeps, s.anderson_history) > 0 &&
        0 < s.mixing <= 1 || throw(ArgumentError("Invalid SCF iteration/mixing settings"))
    s.green_polish_every >= 0 || throw(ArgumentError("Negative polish interval"))
    for key in (
        :energy_tolerance,
        :eigen_tolerance,
        :density_tolerance,
        :green_shift,
        :anderson_regularization,
        :anderson_coefficient_limit,
        :tangent_relative_cutoff,
        :tangent_absolute_cutoff,
        :metric_relative_cutoff,
        :metric_absolute_cutoff,
        :constraint_relative_cutoff,
        :constraint_absolute_cutoff,
        :orthogonalize_threshold,
        :independence_threshold,
        :gram_tolerance,
    )
        value = getfield(s, key)
        isfinite(value) && value > 0 || throw(ArgumentError("Invalid SCF $key"))
    end
    isfinite(s.green_polish_residual) && s.green_polish_residual >= 0 ||
        throw(ArgumentError("Invalid polish residual"))
    s
end
function prepare_run(output, kind, p; options...)
    output === nothing && return nothing
    dir = abspath(output)
    ispath(dir) && throw(ArgumentError("Run output already exists: $dir; choose a fresh directory"))
    mkpath(joinpath(dir, "snapshots"))
    files = sort([
        joinpath(root, f) for (root, _, fs) in walkdir(@__DIR__) for f in fs if endswith(f, ".jl")
    ])
    hashes = Dict(relpath(f, @__DIR__) => bytes2hex(sha256(read(f))) for f in files)
    for f in files
        dest = joinpath(dir, "source", relpath(f, @__DIR__))
        mkpath(dirname(dest))
        cp(f, dest)
    end
    for name in ("Project.toml", "Manifest.toml")
        src = joinpath(@__DIR__, "..", name)
        isfile(src) && cp(src, joinpath(dir, name))
    end
    config = Dict(
        "kind" => kind,
        "problem" => problem_descriptor(p),
        "options" => Dict(string(k) => option_dict(v) for (k, v) in options),
        "started_utc" => string(now(UTC)),
        "julia_version" => string(VERSION),
        "cpu" => Sys.CPU_NAME,
        "blas_threads" => BLAS.get_num_threads(),
        "source_sha256" => hashes,
        "timing_scope" => "in-process end-to-end wall time; includes initialization, diagnostics and checkpoint I/O; excludes Julia process startup and later postprocessing",
    )
    write_json(joinpath(dir, "config.json"), config)
    dir
end
write_json(path, data) = open(io -> JSON3.write(io, data), path, "w")
run_io(dir, name) = dir === nothing ? devnull : open(joinpath(dir, name), "w")
close_run_io(io) = io === devnull ? nothing : close(io)
function fail_run(dir, e)
    dir === nothing ||
        write_json(joinpath(dir, "status.json"), (status = "failed", error = sprint(showerror, e)))
end
