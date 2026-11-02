# 3D density mixing from residual history; reject nonfinite or negative extrapolations.

function density_mix(rho, out, inputs, errors; mixing = 0.2, history = 6)
    residual = out - rho
    linear = rho + mixing * residual
    push!(inputs, copy(linear))
    push!(errors, copy(residual))
    while length(inputs) > history
        popfirst!(inputs)
        popfirst!(errors)
    end
    length(inputs) < 2 && return linear
    B = [dot(x, y) for x in errors, y in errors]
    B ./= max(maximum(abs, B), eps())
    B += 1e-12I
    coefficients = B \ ones(length(inputs))
    den = sum(coefficients)
    isfinite(den) && abs(den) > 1e-12 || return linear
    coefficients ./= den
    maximum(abs, coefficients) <= 20 || return linear
    candidate = sum(coefficients[i] .* inputs[i] for i in eachindex(inputs))
    all(isfinite, candidate) && minimum(candidate) >= 0 ? candidate : linear
end
