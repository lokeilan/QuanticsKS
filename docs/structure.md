# 代码结构

入口顺序是 `run.jl → QuanticsKS.main → configuration_plan / run_config`。配置只在 `configuration.jl` 中转换成模型和选项；数值求解器不读取系统名称或旧项目格式。

| 文件 | 职责 |
|---|---|
| `src/QuanticsKS.jl` | 包入口、显式加载顺序和导出接口 |
| `src/models.jl` | 一维网格、物理模型、占据和单位 |
| `src/options.jl`、`drives.jl` | 数值预算、传播选项和时间外场 |
| `src/states.jl`、`checkpoints.jl` | 轨道状态、初态、细化和数据检查点 |
| `src/hamiltonian.jl` | 密度、静态算符、有效势与总能量 |
| `src/scf.jl` | SCF 密度反馈与停止；审计和输出 Ritz 分文件实现 |
| `src/dynamics.jl` | 分步时间传播 |
| `src/spectrum/` | 固定势求解的选项、诊断、修正与迭代 |
| `src/observables.jl` | 观测量与保存状态后处理 |
| `src/runs.jl` | 选项校验、记录器、源码快照与运行状态 |
| `src/configuration.jl`、`cli.jl` | 共用配置解析、预检与命令行 |
| `src/numerics.jl` | Numerics 模块的唯一加载入口 |

`numerics/` 按计算职责组织：张量和 MPO、点求值与缓存、TCI 和 pivot、线性表达式、切空间、系数问题及 LRRAP。加载关系集中在两个模块入口；求解器文件只定义计算行为。

`parallel.jl` 管理有界独立工作和线程预算；`recorded_parallel.jl` 管理独立拟合的随机流、记录及 pivot 归并。`tensor.jl` 的内积在直接循环和 TCI/BLAS 收缩间选择，内部并行仅在工作量足够且没有外层并行时启用。

`test/` 按张量、算符、场、状态、固定谱、配置和并行行为分文件。独立 dense 矩阵只在验证中使用。`scripts/` 仅保留当前一维实现可使用的微基准与首轮容差实验，不包含旧服务器部署和迁移脚本。
