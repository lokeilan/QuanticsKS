# 1D density history and Pulay mixing, with linear mixing as the fallback.

mutable struct DensityMixer
    inputs::Vector{Vector{Float64}}
    residuals::Vector{Vector{Float64}}
end
DensityMixer() = DensityMixer(Vector{Float64}[], Vector{Float64}[])
function mix_density!(rho, out, c, mixer)
    c.mixing_method == "linear" && return (@. rho = (1 - c.mixing) * rho + c.mixing * out)
    push!(mixer.inputs, copy(rho))
    push!(mixer.residuals, out - rho)
    if length(mixer.inputs) > c.pulay_history
        popfirst!(mixer.inputs)
        popfirst!(mixer.residuals)
    end
    m = length(mixer.inputs)
    m == 1 && return (@. rho = (1 - c.mixing) * rho + c.mixing * out)
    gram = [dot(mixer.residuals[i], mixer.residuals[j]) for i = 1:m, j = 1:m]
    scale = maximum(abs, diag(gram))
    scale > 0 || return copyto!(rho, out)
    gram ./= scale
    for j = 1:m
        gram[j, j] += 1e-12
    end
    system = zeros(m + 1, m + 1)
    system[1:m, 1:m] .= gram
    system[1:m, end] .= 1
    system[end, 1:m] .= 1
    rhs = zeros(m + 1)
    rhs[end] = 1
    coefficients = (system \ rhs)[1:m]
    candidate = zeros(length(rho))
    if all(isfinite, coefficients) && maximum(abs, coefficients) <= 20
        for j = 1:m
            @. candidate += coefficients[j] * (mixer.inputs[j] + c.mixing * mixer.residuals[j])
        end
    else
        fill!(candidate, NaN)
    end
    if all(isfinite, candidate) && minimum(candidate) >= 0
        copyto!(rho, candidate)
    else
        # A failed extrapolation returns to ordinary mixing, without changing
        # occupations, enforcing symmetry, or rescaling the output density.
        @. rho = (1 - c.mixing) * rho + c.mixing * out
        empty!(mixer.inputs)
        empty!(mixer.residuals)
    end
    return rho
end
