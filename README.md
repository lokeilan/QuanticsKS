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
rank, and tolerance convergence. SCF stopping and the final physical audit are
reported separately.

## Use from Julia

```julia
using QuanticsKS

problem = hydrogen_chain(R=6, natoms=4)
result = ground_state(problem; output="results/h4-library")
state = load_checkpoint("results/h4-library/checkpoint.jls")
observables = observe(state)
postprocess("results/h4-library")
```

`postprocess` writes observables, sampled density, and bond profiles into the run
directory. `fixed_spectrum`, `propagate`, and `refine_state` use the same problem
and orbital-state types.

Select an orbital solver with `ground_state(problem; solver=:block_davidson)`.
The choices are `:lrrap` (default), `:lanczos`, `:davidson`, and
`:block_davidson`. The three subspace methods are also available as
`fixed_spectrum(state, potential; solver=:lanczos, spectrum=options)`.
Lanczos uses KrylovKit's scalar or block three-term recurrence, with
reorthogonalization and explicit restart from the retained Ritz orbitals.
The Davidson choices use generalized Davidson with a shifted kinetic-inverse
preconditioner and share KrylovKit's direction orthogonalization.

## Documentation

- [Algorithm](docs/algorithm.md): discretization, SCF, tensor operations, and propagation.
- [Configuration](docs/configuration.md): model inputs, numerical controls, and output files.
- [Structure](docs/structure.md): entry point, shared functions, and source layout.
