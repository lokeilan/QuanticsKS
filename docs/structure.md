# Structure

The organization follows the separation of calculation scripts, shared functions,
and observables in
[Gross-Pitaevskii-TCI](https://github.com/MarcelNiedermeier/Gross-Pitaevskii-TCI).
QuanticsKS uses one `run.jl` entry point and selects the physical system through
an input file. The SCF loop is shared by 1D and 3D models.

```text
run.jl
Project.toml
Manifest.toml
README.md
inputs/
    h4.toml
    helium.toml
    propagation.toml
docs/
    algorithm.md
    configuration.md
    structure.md
src/
    QuanticsKS.jl
    configuration.jl
    models.jl
    states.jl
    hamiltonian.jl
    scf.jl
    spectrum.jl
    dynamics.jl
    observables.jl
    tensor.jl
    tci.jl
    physics.jl
    solvers.jl
```

## Calculation flow

`run.jl` calls `QuanticsKS.main`. The configuration parser builds a `Problem`
and numerical options, then calls `ground_state` or `propagate`. Model-specific
methods supply the grid, fields, kinetic operator, and Green operation.

| File | Responsibility |
|---|---|
| [QuanticsKS.jl](../src/QuanticsKS.jl) | Package imports, source loading, and exports |
| [configuration.jl](../src/configuration.jl) | TOML parsing, validation, command line, and run output |
| [models.jl](../src/models.jl) | 1D and 3D grids, physical models, occupations, and options |
| [states.jl](../src/states.jl) | Initial orbitals, refinement, checkpoints, and 1D low-pass filtering |
| [hamiltonian.jl](../src/hamiltonian.jl) | 1D and 3D fields, Hamiltonian operators, Green solves, and total energy |
| [scf.jl](../src/scf.jl) | Shared SCF loop, output Ritz rotation, and physical audit |
| [spectrum.jl](../src/spectrum.jl) | Fixed-potential eigensolvers and residual diagnostics for 1D |
| [dynamics.jl](../src/dynamics.jl) | Split-step time propagation for 1D |
| [observables.jl](../src/observables.jl) | State observables and saved-run postprocessing |

## Shared numerical functions

`QuanticsKS.Numerics`, abbreviated internally as `Q`, contains the tensor
operations used by every model. Its files are loaded by `QuanticsKS.jl`.

| File | Responsibility |
|---|---|
| [tensor.jl](../src/tensor.jl) | QTT algebra, MPO operations, linear expressions, and compression recovery |
| [tci.jl](../src/tci.jl) | Sample caches, TCI fitting, pivot reuse, fit records, and task scheduling |
| [physics.jl](../src/physics.jl) | Periodic soft-Coulomb kernel and scalar 1D XC formulas |
| [solvers.jl](../src/solvers.jl) | LRRAP, restarted Lanczos and Davidson updates, shared orbital algebra, mixing, and storage |

TensorCrossInterpolation supplies fitting, tensor addition, compression, and
contraction environments, including the signed 3D kernel suffix. ITensorMPS
supplies MPO matrix elements, canonical forms, compression recovery, and ordinary
MPO application through TCI's conversion extension.
Hamiltonian projection reuses converted orbitals and operators within a call.
Checkpoints store QTT cores and retain their package type identities.

Local tensor loops implement the grid's carry and borrow rules, bounded sample
caches, and cancellation-sensitive expressions. Dependent SCF, Green, and
tangent-coordinate updates remain explicit in the solver.

Read `scf.jl` for the ground-state workflow, `hamiltonian.jl` for its physical
operators, and `solvers.jl` for orbital updates and their shared numerical
functions. Dimension-specific methods sit with the operation they implement.
