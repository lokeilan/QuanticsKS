# QuanticsKS

QuanticsKS solves Kohn-Sham ground states with quantized tensor trains (QTT) and
tensor cross interpolation (TCI). It supports periodic one-dimensional
soft-Coulomb systems and isolated three-dimensional all-electron systems. The 1D
solver also supports fixed-potential spectra and real-time propagation.

## Run an example

Use Julia 1.12. `Manifest.toml` fixes the dependency versions.

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. run.jl plan inputs/h4.toml
julia --threads=4,1 --project=. run.jl solve inputs/h4.toml results/h4
```

`plan` validates the input and prints the grid and numerical settings without
constructing orbitals. `solve` requires a new output directory.

| Input | Calculation |
|---|---|
| [h4.toml](inputs/h4.toml) | Periodic 1D hydrogen chain with four atoms |
| [helium.toml](inputs/helium.toml) | Isolated 3D helium on a coarse grid |
| [propagation.toml](inputs/propagation.toml) | Short 1D evolution in a cosine potential |

These examples demonstrate the workflow. Physical accuracy requires grid, box,
rank, and tolerance convergence. A saved SCF state is accepted only when its independent physical checks pass.

The [periodic 1D H100 input](inputs/h100-r16.toml) defines the current baseline
at R16; change `[grid].R` for another resolution. It uses 50 occupied orbitals
and six auxiliary empty states, with orbital, algebra, orthogonalization and
storage tolerances of `1e-12`, and orbital, work and Green right-hand-side caps
of 64. The inner solve follows
the preceding occupied space within numerically unresolved boundary clusters.
The density update uses a constant-susceptibility dielectric inverse before
Anderson mixing. Acceptance requires a relative density residual of `1e-6`,
an orbital residual below `3e-4 Ha` including its action-error budget, and
consistent Gram, charge, and independently recomputed energy. The general
`scf.extra_orbitals` default remains zero. An occupied-space Rayleigh–Ritz
rotation produces the saved orbitals. See the
[configuration guide](docs/configuration.md#periodic-1d-h100-profile).

## Use from Julia

```julia
using QuanticsKS

problem = hydrogen_chain(R=6, natoms=4)
result = ground_state(problem; output="results/h4-library")
println("Accepted: ", result.accepted)
state = load_checkpoint("results/h4-library/checkpoint.jls")
observables = observe(state)
postprocess("results/h4-library")
```

`postprocess` writes observables, sampled density, and bond profiles into the run
directory. `fixed_spectrum`, `propagate`, and `refine_state` use the same problem
and orbital-state types.

Select an orbital solver with `ground_state(problem; solver=:block_davidson)`.
The choices are `:lrrap` (default), `:block_lanczos`, and `:block_davidson`.
The subspace methods are also available as
`fixed_spectrum(state, potential; solver=:block_lanczos, spectrum=options)`.
Block Lanczos uses KrylovKit's block three-term recurrence, including for a
single orbital, with reorthogonalization and explicit restart from the retained
Ritz orbitals. Block generalized Davidson uses a shifted kinetic-inverse
preconditioner and KrylovKit's direction orthogonalization.

LRRAP also supports full-grid dense orbitals:

```julia
result = ground_state(problem; representation=:dense, solver=:lrrap)
```

Dense and QTT LRRAP share the SCF loop, constrained coefficient solve, Green
updates, occupation selection, and final physical audit. Dense directions use
the full grid space; QTT directions use the current TT tangent space. The
dense path uses SparseArrays, FFTW, and LinearAlgebra without TCI fits. Set
`representation = "dense"` in `[run]` to select it from TOML. Dense support is
limited to LRRAP ground-state SCF in 1D and 3D.

## Documentation

- [Algorithm](docs/algorithm.md): discretization, SCF, tensor operations, and propagation.
- [Configuration](docs/configuration.md): model inputs, numerical controls, and output files.
- [Structure](docs/structure.md): entry point, shared functions, and source layout.
