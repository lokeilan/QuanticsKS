# 代码结构

3D extension: [models, parameters, numerical conventions, and limitations](three-dimensional.md). `src/models3d.jl` defines geometry and models; `src/three_dimensional.jl` supplies Cartesian operators and adapters to the shared SCF workflow.


从 `run.jl → QuanticsKS.main → run_config` 进入计算。输入在 `configuration.jl` 中转换成模型和选项，随后进入 SCF、固定势本征求解或传播。氢链、一般软库仑模型和局域势共用这些流程。

活动源码的加载顺序集中在 `src/QuanticsKS.jl`。同一计算步骤的选项、辅助函数和执行循环放在一起。

| 文件 | 职责 |
|---|---|
| `src/QuanticsKS.jl` | 包入口、Numerics 模块和导出接口 |
| `src/models.jl` | 网格、物理模型、占据、选项和时间外场 |
| `src/models3d.jl` | 三维网格、坐标编码和模型输入 |
| `src/states.jl` | 初态、网格细化、检查点和独立低通操作 |
| `src/hamiltonian.jl` | 外势、密度、Hartree/XC 有效势及总能量 |
| `src/scf.jl` | SCF 循环、输出 Ritz 旋转和物理审计 |
| `src/spectrum.jl` | 固定势本征求解、子空间更新和诊断 |
| `src/dynamics.jl` | 势能半步和动能整步的时间传播 |
| `src/observables.jl` | 观测量和保存状态后处理 |
| `src/configuration.jl` | 配置、预检、运行记录和命令行 |
| `src/three_dimensional.jl` | 三维算子、边界条件和求解接口扩展 |
| `src/numerics/tensor.jl` | TT 代数、点求值、线性表达式和 MPO |
| `src/numerics/sampling.jl` | 有界缓存、点值和批量采样表达式 |
| `src/numerics/tci.jl` | TCI 拟合、pivot 复用、计时和记录归并 |
| `src/numerics/physics.jl` | 周期软库仑核及 1D LSDA 标量公式 |
| `src/numerics/hamiltonian.jl` | Hamiltonian 作用、切空间和投影 |
| `src/numerics/lrrap.jl` | 轨道基变换、系数问题、LRRAP、密度混合和存储舍入 |
| `src/numerics/parallel.jl` | 有界任务调度 |
| `src/numerics/svd_safety.jl` | 库压缩和数值失败后的诊断恢复 |

物理计算采用 `psi` / `orbitals` 表示轨道，`rho` 表示密度，`Vext`、`Vhartree`、`Vxc`、`Veff` 表示势，`dx` 表示网格间距。`R` 是二进制位数，`L` 是周期长度；矩阵分解中的 `U`、`S`、`V` 保留常用数学记法。轨道核心保存的是 `sqrt(dx)*psi`，密度和积分只在相应物理步骤处理 `dx`。

组织方式参考 [Gross-Pitaevskii-TCI](https://github.com/MarcelNiedermeier/Gross-Pitaevskii-TCI) 的主流程、物理量命名和直接库调用，以及 [tensor4all 的 Quantics 示例](https://github.com/tensor4all/T4APlutoExamples/tree/main/pluto_notebooks)。参考项目的 GPE 模型和谱动能不用于替换本项目的 KS/FD2 定义。

常规相加、压缩、内积和端点环境调用 TensorCrossInterpolation。MPO 矩阵元、无截断规范化及默认 MPO 作用调用 ITensorMPS；两种表示通过 TCI 自带的转换扩展衔接，检查点仍保存原来的 QTT 核。Hamiltonian 投影在一次调用内复用已转换的轨道和算子，串行与并行共用同一实现。

本地保留切空间坐标投影、物理 MPO 构造、有界采样缓存、统计和失败诊断。三维 Green 的截断 CG 与坐标编码仍有独立的库迁移空间。当前后端与三维核对见[ITensorMPS 迁移记录](itensormps-3d-20261108.md)；前一轮判断保存在[循环核查](loop-audit-20261107.md)。

`test/physics_consistency.jl` 使用独立镜像求和、交换积分和相关能导数，核对场、Hamiltonian 作用及能量。生产求解器不依赖测试中的 dense 数组，也不依赖本地历史源码目录。
