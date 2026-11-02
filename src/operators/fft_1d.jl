# FFT workspace, kinetic preconditioning, and Hartree convolution with one grid weight.

"""Full complex FFTs on N points. No spatial or reflection-sector reduction."""
struct FFTWorkspace{F,B}
    buffer::Vector{ComplexF64}
    forward::F
    backward::B
end
function FFTWorkspace(n)
    a = zeros(ComplexF64, n)
    FFTWorkspace(a, plan_fft!(a; flags = FFTW.ESTIMATE), plan_ifft!(a; flags = FFTW.ESTIMATE))
end
struct KineticPreconditioner{W}
    fft::W
    denominator::Vector{Float64}
    counter::Counter
end
function KineticPreconditioner(n, dx, shift)
    KineticPreconditioner(
        FFTWorkspace(n),
        [shift + 2sinpi(j / n)^2 / dx^2 for j = 0:(n - 1)],
        Counter(),
    )
end
function LinearAlgebra.ldiv!(Y::AbstractVecOrMat, p::KineticPreconditioner, X::AbstractVecOrMat)
    t0 = time_ns()
    size(Y) == size(X) || throw(DimensionMismatch())
    w = p.fft
    @inbounds for b in axes(X, 2)
        for i in axes(X, 1)
            w.buffer[i] = X[i, b]
        end
        w.forward * w.buffer
        w.buffer ./= p.denominator
        w.backward * w.buffer
        for i in axes(X, 1)
            Y[i, b] = real(w.buffer[i])
        end
    end
    p.counter.calls += 1
    p.counter.columns += size(X, 2)
    p.counter.seconds += (time_ns() - t0) * 1e-9
    return Y
end

struct HartreeWorkspace{W}
    fft::W
    kernel_fft::Vector{ComplexF64}
    dx::Float64
end
function hartree!(v, h::HartreeWorkspace, rho)
    w = h.fft
    w.buffer .= rho
    w.forward * w.buffer
    w.buffer .*= h.kernel_fft
    w.backward * w.buffer
    @. v = h.dx * real(w.buffer)
    return v
end
