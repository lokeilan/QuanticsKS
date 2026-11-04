# Time-dependent local scalar potentials and their validation.

abstract type AbstractDrive end
struct NoDrive <: AbstractDrive end
"Periodic scalar pulse with a sin² envelope; amplitude is a potential energy in Ha."
Base.@kwdef struct CosinePulse <: AbstractDrive
    amplitude::Float64 = 0.01
    start::Float64 = 0.0
    duration::Float64 = 1.0
    frequency::Float64 = 0.0
    phase::Float64 = 0.0
    spatial_mode::Int = 1
    spatial_phase::Float64 = 0.0
end
struct FunctionDrive{F} <: AbstractDrive
    potential::F
    id::String
end
drive_value(::NoDrive, x, t, L) = 0.0
function drive_value(d::CosinePulse, x, t, L)
    tau = t - d.start
    0 < tau < d.duration || return 0.0
    d.amplitude *
    sinpi(tau / d.duration)^2 *
    cos(d.frequency * tau + d.phase) *
    cos(2pi * d.spatial_mode * x / L + d.spatial_phase)
end
function drive_value(d::FunctionDrive, x, t, L)
    v = d.potential(x, t)
    isreal(v) && isfinite(v) ||
        throw(ArgumentError("Drive must return finite real potential energies"))
    real(v)
end
drive_descriptor(::NoDrive) = Dict("kind" => "none")
drive_descriptor(d::CosinePulse) = merge(Dict("kind" => "cosine_pulse"), option_dict(d))
drive_descriptor(d::FunctionDrive) = Dict("kind" => "function", "id" => d.id)
function validate(d::AbstractDrive)
    if d isa CosinePulse
        d.duration > 0 && all(
            isfinite,
            (d.amplitude, d.start, d.duration, d.frequency, d.phase, d.spatial_phase),
        ) || throw(ArgumentError("Invalid pulse"))
    elseif d isa FunctionDrive
        isempty(d.id) && throw(ArgumentError("Drive id is required"))
    end
    d
end
