# QuanticsKS

一维周期 Kohn–Sham 计算的 Julia 包。轨道和场使用 QTT 表示，TCI 负责采样重构；支持低秩 SCF、固定势本征谱、实时传播、网格细化和保存状态的后处理。

仓库只有一套活动实现：`src/QuanticsKS.jl`。氢链、软库仑模型和局域余弦势共用同一计算流程。旧 TCI-TDKS 外壳、硅和生产 dense 求解器已移出活动代码；小网格 dense 运算仅作为数值验证的独立参照。

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

R16 输入是较重的计算。它不代表三维 H100 文献复现。当前正式容差保持密度/XC `1e-10`、TCI 后置 SVD 系数 `0.1`。

SCF 命令退出码 `0` 表示达到配置的停止条件，`2` 表示迭代完成但未达到停止条件，`1` 表示输入或执行错误。能量停止和独立物理审计分别输出；达到能量判据不能代替轨道残差与密度接受。传播成功完成配置步数返回 `0`。

## 并行

启动时用 `--threads` 分配 Julia 工作线程。SCF 在 `[scf]`、传播在 `[time]` 设置 `parallel_tasks=0` 使用可用工作线程；`1` 为串行，正整数限制并发数。默认仍为 `1`，`[run].blas_threads` 建议在轨道并行时使用 `1`。

独立轨道拟合、投影、Green 更新、Gram/Ritz、存储和审计可并行。单个较大的直接内积也可内部并行，小收缩保持串行；外层任务内不再次拆分。SCF 轮次、依赖最新系数的扫描和 TT site 递推保持顺序。日志、随机流和 pivot 状态按固定顺序归并。

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
- [整理记录](docs/migration.md)：移除范围、保留的历史证据及验证范围。

```powershell
julia --startup-file=no --project=. test/runtests.jl
```

回归覆盖独立 dense 对照、张量代数、TCI/pivot、MPO、SCF、固定势、传播、重启，以及单工作线程和多工作线程行为。回归成功不表示生产规模物理收敛。
