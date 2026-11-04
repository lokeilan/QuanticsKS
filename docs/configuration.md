# 配置与输出

TOML 使用唯一的一维格式，不再接受 `[calculation]` 分发段。`[system].kind` 支持 `hydrogen_chain`、`soft_coulomb` 和 `cosine_potential`。示例见 `inputs/`；未知选项会报错。

`[run].kind` 为 `scf` 或 `propagation`。SCF 使用 `[scf]`，传播使用 `[time]` 和可选 `[drive]`；其余可用段落为 `[grid]`、`[accuracy]`、`[tci]`、`[guess]`、`[physics]`、`[audit]`、`[refinement]`。

| 设置 | 含义 |
|---|---|
| `accuracy.field_tolerance` | 势场构建精度 |
| `accuracy.work_tolerance` | 轨道修正工作张量预算 |
| `accuracy.storage_tolerance` | 保存轨道精度 |
| `accuracy.density_tolerance` / `xc_energy_tolerance` | 密度及 XC 能量密度的 TCI 容差，默认均为 `1e-10` |
| `tci.post_rounding_factor` | 已有 TCI 后置 SVD 容差系数，默认 `0.1` |
| `scf.criterion` | `energy` 或需额外接受条件的 `strict` |
| `scf.energy_tolerance` | 相邻轮全体系总能量变化阈值，默认 `1e-6 Ha` |
| `scf.parallel_tasks` / `time.parallel_tasks` | 独立工作并发预算 |

`[run].checkpoint` 相对于输入文件目录解析。它导入轨道，SCF 历史从第一轮重新开始；同目录有效的 `cross-state.jls` 可用于 pivot 热启动。细化目标网格时使用 `[refinement]`。输出路径相对于启动目录；命令行 `NEW_OUTPUT` 优先于输入中的 `run.output`。

每次运行保存 `input.toml`、`config.json`、`status.json`、迭代日志和数据检查点。`workflow.json` 记录配置运行的时间范围；`source/` 保存完整包的 `src/`、依赖声明及 `run.jl`。可以将 `source/` 作为 Julia 项目启动，并以保存的输入进行新的计算；重启所引用的外部检查点须继续可用。

SCF 的 `scf.jsonl` 每行是一轮墙钟时间，初始化、检查点写入及末尾审计不都包含在这一行内。并行子任务时间可能重叠，不能相加后当作整轮墙钟时间。轨道核存储、累计分配和峰值内存是不同量。

能量停止与物理接受分别保存在 `status.json` 的 `converged` 与 `audit.passed`。`criterion=energy` 可以在物理审计未通过时停止；首轮没有相邻能量差。命令行同时显示停止结果和审计结果，不合并成含义不明的“已接受”。
