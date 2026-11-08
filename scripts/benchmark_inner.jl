# Measure the library contraction on synthetic and saved orbitals.
using QuanticsKS, LinearAlgebra, Random, JSON3, Statistics, Serialization
const Q = QuanticsKS.Q
length(ARGS) == 2 || error("Usage: benchmark_inner.jl NEW_OUTPUT.json CHECKPOINT")
output = abspath(ARGS[1])
ispath(output) && error("Output already exists: $output")
BLAS.set_num_threads(1)
rng = MersenneTwister(20261106)

function random_orbital(rng, R, cap)
    bonds = [min(2^min(k, R - k), cap) for k = 0:R]
    Q.normalize(
        Q.QTT([randn(rng, ComplexF64, bonds[k], 2, bonds[k + 1]) / sqrt(2bonds[k]) for k = 1:R]),
    )
end

# Contraction timings need only saved amplitudes, including function-defined models.
orbitals = [Q.QTT(cores) for cores in deserialize(ARGS[2])["cores"]]
cases = [
    ("synthetic_cap_$cap", random_orbital(rng, 16, cap), random_orbital(rng, 16, cap)) for
    cap in (16, 32, 48)
]
norb = length(orbitals)
for i in unique([1, cld(norb, 2), norb])
    push!(cases, ("saved_orbital_$i", orbitals[i], orbitals[mod1(i + 1, norb)]))
end
results = map(cases) do (name, bra, ket)
    expected = Q.inner(bra, ket)
    samples = map(1:8) do _
        GC.gc()
        @timed Q.inner(bra, ket)
    end
    (;
        name,
        median_seconds = median(sample.time for sample in samples),
        allocated_bytes = maximum(sample.bytes for sample in samples),
        seconds = [sample.time for sample in samples],
        repeated_value_max_difference = maximum(abs(sample.value - expected) for sample in samples),
    )
end
report = (;
    scope = "Exact TCI inner contraction; warmed calls include allocation and any GC, exclude process startup. No local inner backend remains.",
    julia_threads = Threads.nthreads(:default),
    blas_threads = BLAS.get_num_threads(),
    results,
)
mkpath(dirname(output))
open(io -> JSON3.pretty(io, report), output, "w")
println("Saved contraction timings to ", output)
