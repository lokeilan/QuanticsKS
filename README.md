# QuanticsKS

QuanticsKS is a Julia QTT/TCI solver for one-dimensional periodic and three-dimensional isolated Kohn-Sham ground states. The 1D model also supports fixed-potential spectra and real-time propagation.

The active implementation is `src/QuanticsKS.jl`. Both dimensions reuse the same tensor algebra, LRRAP updates, and SCF workflow. See [three-dimensional models and paper parameters](docs/three-dimensional.md) for the 3D conventions, runnable inputs, and limitations. Small-grid dense calculations serve only as independent validation.

## 运行

使用 Julia 1.12，依赖版本由 `Manifest.toml` 锁定：

```powershell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=. run.jl plan inputs/h4.toml
julia --startup-file=no --threads=4,1 --project=. run.jl solve inputs/h4.toml results/h4
```

`plan` 只校验配置、列出网格与线程设置，不构造轨道或场。`solve` 要求新的输出目录，不覆盖已有结果。

| 输入 | 用途 |
|---|---|
| `inputs/h4.toml` | 小规模氢链 SCF，启用 Julia 任务并行 |
| `inputs/h100-r16.toml` | 一维周期 100-H 内部模型，R16、50 条双占据轨道 |
| `inputs/propagation.toml` | 小网格局域余弦势的实时传播 |

R16 输入是较重的计算。它不代表三维 H100 文献复现。密度/XC 默认容差为 `1e-10`。TCI 默认保留 prrLU 构造结果；只有显式设置正的 `post_rounding_factor` 才追加 SVD 舍入。

SCF 命令退出码 `0` 表示达到配置的停止条件，`2` 表示迭代完成但未达到停止条件，`1` 表示输入或执行错误。能量停止和独立物理审计分别输出；达到能量判据不能代替轨道残差与密度接受。传播成功完成配置步数返回 `0`。

## 并行

启动时用 `--threads` 分配 Julia 工作线程。SCF 在 `[scf]`、传播在 `[time]` 设置 `parallel_tasks=0` 使用可用工作线程；`1` 为串行，正整数限制并发数。默认仍为 `1`，`[run].blas_threads` 建议在轨道并行时使用 `1`。

独立轨道拟合、投影、Green 更新、Gram/Ritz、存储和审计可并行。内积直接调用 TCI 收缩，任务并行放在轨道和矩阵层。SCF 轮次、依赖最新系数的扫描和 TT site 递推保持顺序。日志、随机流和 pivot 状态按固定顺序归并。

## Julia 接口

```julia
using QuanticsKS
p = hydrogen_chain(R=6, natoms=4)
result = ground_state(p; scf=SCFOptions(parallel_tasks=0), output="results/library-h4")
state = load_checkpoint("results/library-h4/checkpoint.jls")
observe(state; parallel_tasks=0)
```

`fixed_spectrum`、`propagate`、`refine_state`、`postprocess` 使用相同的模型、状态与精度类型。检查点保留数据 schema 和 QuanticsKS 类型身份，路径迁移不要求转换现有一维检查点。

## 阅读与验证

- [代码结构](docs/structure.md)：模块职责和阅读顺序。
- [配置与输出](docs/configuration.md)：输入、重启、源码快照和停止含义。
- [算法流程](docs/algorithm.md)：构建、秩控制、投影和消费者之间的关系。
- [本次重构与物理核对](docs/refactor-20261106.md)：库复用、结构和 XC 修正。
- [循环与软件包复用核查](docs/loop-audit-20261107.md)：直接替换、库迁移候选和保留循环的职责。
- [ITensorMPS 迁移与 3D 核对](docs/itensormps-3d-20261108.md)：当前张量后端、修正和数值对照。
- [整理记录](docs/migration.md)：移除范围、保留的历史证据及验证范围。

```powershell
julia --startup-file=no --project=. test/runtests.jl
```

回归覆盖独立 dense 对照、张量代数、TCI/pivot、MPO、SCF、固定势、传播、重启，以及单工作线程和多工作线程行为。回归成功不表示生产规模物理收敛。
