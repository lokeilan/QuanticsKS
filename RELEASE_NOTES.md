# V1 (dedup) — v1.0.0

Publishes registered version `v1-dedup-20261103`. Numerical source commit: `c00c58145c2268ad66dbcca1bb8d01aa5d987329`, also tagged `milestone/r16-dedup-20261103`. Publication documentation is added on top of this numerical snapshot.

V1 reduces repeated sampling setup and Gram contractions, shares read-only pivot snapshots and fixed-H potential samples, and retains pivot warm starts, sampled Ritz projection and combined orbital reconstruction.

## Recorded R16 cold start

Model: One-dimensional periodic soft-Coulomb H100 internal model with unpolarized 1D LSDA. The grid has 65536 points and 50 doubly occupied orbitals. The recorded run used Julia 1.12.5, 8 Julia threads and 8 BLAS threads.

| Measurement | Recorded value |
|---|---:|
| SCF iterations | 13 |
| Energy-change stopping tolerance (Ha) | 1e-06 |
| Final energy change (Ha) | 1.526276492e-07 |
| Final total energy (Ha) | -66.14298211 |
| Maximum stored orbital rank | 30 |
| SCF iteration time sum (s) | 1132.699925 |
| Complete Julia process wall time (s) | 1262.208678 |
| Audit maximum single-orbital residual (Ha) | 0.000693148826 |
| Audit relative density residual | 0.0003479694287 |
| Audit Gram defect | 8.820182827e-12 |

The energy-change criterion was reached. Independent physical audit passed: **false**. The state is not certified as physically converged.

SCF time is the sum of iteration timers. Complete-process time also includes startup, initialization, output and final audit. These timings describe one archived local run.

## Verification and scope

The V1 library regressions and forced fixed-H Ritz validation completed in the recorded campaign. The publication checkout is separately checked for package loading, R16 input planning and library regression. These checks do not replace physical acceptance.

Energy stopping was reached, but independent physical acceptance failed. This is one local R16 measurement and does not establish performance for other grids or reproduction of a three-dimensional H100 calculation.

The release contains source, tests, pinned dependencies, the R16 input and publication documentation. Archived outputs, checkpoints and later working-tree changes are excluded. GitHub supplies the source download for this tag.

Recorded measurement provenance: Recorded V1 run: h100-r16-dedup-local-20261103-v1; run/status.json, run/scf.jsonl and process.json.
