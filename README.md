# TCI-TDKS-3RD — V1 (dedup)

QTT/TCI Kohn–Sham research code. This release publishes the registered V1 numerical implementation, `v1-dedup-20261103`, from commit `c00c58145c2268ad66dbcca1bb8d01aa5d987329`. The release commit adds publication documentation; the numerical source, tests, dependency declarations and R16 input retain their V1 contents.

V1 reuses pivot geometry and sampled values, avoids unnecessary TTCache setup when endpoint environments cover the grid, shares fixed-H potential samples, and reuses Gram contractions. It includes sampled Ritz projection and combined orbital transformations.

## Requirements and execution

Use Julia 1.12. The recorded R16 run used Julia 1.12.5, eight Julia threads and eight BLAS threads. Install the pinned dependencies from the repository root:

```powershell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Inspect the supplied one-dimensional periodic H100 configuration:

```powershell
julia --startup-file=no --threads=8 --project=. run.jl plan inputs/h100-r16-qtt.toml
```

Run into a directory that does not already exist:

```powershell
julia --startup-file=no --threads=8 --project=. run.jl solve inputs/h100-r16-qtt.toml results/h100-r16-v1
```

The supplied input describes 100 electrons in 50 doubly occupied spatial orbitals, 65,536 grid points and a 400-bohr periodic cell. Orbital cores store `sqrt(h) * psi`. This is the one-dimensional internal H100 model; it does not establish reproduction of a three-dimensional H100 paper calculation.

## Recorded result and numerical acceptance

The archived V1 R16 cold start reached the `1e-6 Ha` energy-change stopping condition after 13 SCF iterations. The independent physical audit failed. Its maximum single-orbital residual was approximately `6.93e-4 Ha`, and its relative density residual was approximately `3.48e-4`.

Energy stopping and independent physical acceptance are recorded separately. Detailed measurements, timing scopes and limitations are in [the release notes](RELEASE_NOTES.md) and [machine-readable release data](docs/releases/v1.json). The timing describes one recorded local run.

## Code and verification

- `run.jl` and `src/`: common input and calculation interface, dense reference and silicon solver implementations.
- `lib/QuanticsKS/src/`: QTT representation, sampling, operators, SCF and fixed-H solvers.
- `lib/QuanticsKS/test/`: algebra, operator, cache, pivot-restart, solver and checkpoint regressions.
- `inputs/h100-r16-qtt.toml`: the registered V1 input.

Run the numerical library regression from the repository root:

```powershell
julia --startup-file=no --threads=1 --project=. lib/QuanticsKS/test/runtests.jl
```

The silicon implementations are retained as part of the V1 source snapshot. Their production inputs, reference datasets and pseudopotential files require separate configuration. Archived calculation outputs, checkpoints, temporary files and later working-tree experiments are outside this release.

To regenerate the release notes from the recorded release data:

```powershell
julia --project=. scripts/release_notes.jl
```
