# Periodic FD2 kinetic and local-potential actions; input and output buffers must differ.

mutable struct Counter
    calls::Int
    columns::Int
    seconds::Float64
end
Counter() = Counter(0, 0, 0.0)
reset!(c::Counter) = (c.calls = 0; c.columns = 0; c.seconds = 0.0; nothing)

struct Hamiltonian
    potential::Vector{Float64}
    dx::Float64
    counter::Counter
end
Base.size(h::Hamiltonian) = (length(h.potential), length(h.potential))
Base.size(h::Hamiltonian, d::Int) = d <= 2 ? length(h.potential) : 1
Base.eltype(::Hamiltonian) = Float64

function kinetic!(Y::AbstractVecOrMat, X::AbstractVecOrMat, dx)
    n = size(X, 1)
    size(Y) == size(X) || throw(DimensionMismatch())
    Base.mightalias(Y, X) && error("kinetic! requires distinct input and output")
    a = 0.5 / dx^2
    @inbounds for b in axes(X, 2)
        Y[1, b] = a * (2X[1, b] - X[n, b] - X[2, b])
        for i = 2:(n - 1)
            Y[i, b] = a * (2X[i, b] - X[i - 1, b] - X[i + 1, b])
        end
        Y[n, b] = a * (2X[n, b] - X[n - 1, b] - X[1, b])
    end
    return Y
end

function LinearAlgebra.mul!(Y::AbstractVecOrMat, h::Hamiltonian, X::AbstractVecOrMat)
    t0 = time_ns()
    kinetic!(Y, X, h.dx)
    @inbounds for b in axes(X, 2), i in axes(X, 1)
        Y[i, b] += h.potential[i] * X[i, b]
    end
    h.counter.calls += 1
    h.counter.columns += size(X, 2)
    h.counter.seconds += (time_ns() - t0) * 1e-9
    return Y
end

function density!(rho, U, no, dx)
    fill!(rho, 0.0)
    @inbounds for b = 1:no, i in eachindex(rho)
        rho[i] += 2abs2(U[i, b]) / dx
    end
    return rho
end
