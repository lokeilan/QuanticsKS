# QuanticsKS

本目录是项目使用的 QTT 数值库。`src/QuanticsKS.jl` 只负责导入依赖、加载文件和导出接口。

- `systems.jl`：加载 `models.jl`（网格、模型、占据数）和 `options.jl`（精度、迭代与审计选项）。
- `states.jl`：轨道状态、初态与网格细化。
- `hamiltonian.jl`：密度、有效哈密顿量与能量。
- `scf.jl`：低秩自洽主循环；独立审计在 `scf_audit.jl`，输出轨道的 Ritz 处理在 `scf_output_ritz.jl`。
- `spectrum/`：固定势本征谱的选项、诊断、更新与求解流程，由 `fixed_spectrum.jl` 加载。
- `dynamics.jl`：时间传播。
- `checkpoints.jl`、`runs.jl`、`observables.jl`：状态保存、计算记录与观测量。
- `configuration.jl`：TOML 配置和运行接口。
- `numerics/`：张量运算、TCI、数值物理和低秩更新。

`numerics/` 内按运算职责分文件，仍共用 `Numerics` 模块：

| 文件 | 职责 |
|---|---|
| `tt.jl` | 加载张量基础文件，保留原 include 入口 |
| `tensor.jl` | QTT 存储、缩放、内积、加法与压缩 |
| `evaluation.jl` | 点求值、端点环境缓存与二进制标签 |
| `sample_cache.jl`、`sampled_ops.jl` | 拟合内点值缓存、交叉切片批量求值与共享轨道采样 |
| `cross_state.jl` | 完整交叉索引的跨迭代保存、恢复与容量控制 |
| `mpo.jl` | MPO 构造、作用与 Green 算符 |
| `tangent.jl` | 正交框架、切空间投影与重建 |
| `svd_safety.jl` | SVD 检查、重试与异常记录 |
| `instrumentation.jl`、`cross.jl` | 计时和计数；TCI 重构与线性组合 |
| `hamiltonian.jl` | 底层 FD2 作用、投影与期望值 |
| `orbital_basis.jl`、`coefficients.jl` | 轨道正交化、Ritz 旋转和小系数问题 |
| `lrrap.jl` | 单次低秩轨道更新的 L1 到 L7 流程 |
| `physics.jl`、`interpolation.jl`、`iteration_support.jl` | 标量物理公式、网格细化、密度混合与输出压缩 |

阅读求解流程可从 `scf.jl` 进入 `numerics/lrrap.jl`，再按需要查看各运算文件。英文注释说明文件职责、归一化、数组含义和关键步骤；函数名和类型所在模块保持不变。

缓存、pivot 热启动与正交化／Ritz 合并重构的有效期、参数和验证范围见[实现说明](../../docs/pivot-sampling-20261102.md)。

从项目根目录使用 `julia --project=. run.jl`。库测试位于 `test/runtests.jl`，项目统一测试入口也会运行该测试集。

本目录从 `vendor/QuanticsKS-profiled-20261031` 拆分整理而来，保留原数值定义。历史包和已保存的计算结果由项目单独保留。
