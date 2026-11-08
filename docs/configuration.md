# 配置与输出

3D extension: [models, parameters, numerical conventions, and limitations](three-dimensional.md). `src/models3d.jl` defines geometry and models; `src/three_dimensional.jl` supplies Cartesian operators and adapters to the shared SCF workflow.


TOML 使用唯一的一维格式，不再接受 `[calculation]` 分发段。`[system].kind` 支持 `hydrogen_chain`、`soft_coulomb` 和 `cosine_potential`。示例见 `inputs/`；未知选项会报错。

`[run].kind` 为 `scf` 或 `propagation`。SCF 使用 `[scf]`，传播使用 `[time]` 和可选 `[drive]`；其余可用段落为 `[grid]`、`[accuracy]`、`[tci]`、`[guess]`、`[physics]`、`[audit]`、`[refinement]`。

| 设置 | 含义 |
|---|---|
| `accuracy.field_tolerance` | 势场构建精度 |
| `accuracy.work_tolerance` | 轨道修正工作张量预算 |
| `accuracy.storage_tolerance` | 保存轨道精度 |
| `accuracy.density_tolerance` / `xc_energy_tolerance` | 密度及 XC 能量密度的 TCI 容差，默认均为 `1e-10` |
| `tci.post_rounding_factor` | 可选的 TCI 后置 SVD 容差系数；默认 `0` 关闭，正值才启用 |
| `scf.criterion` | `energy` 或需额外接受条件的 `strict` |
| `scf.energy_tolerance` | 相邻轮全体系总能量变化阈值，默认 `1e-6 Ha` |
| `scf.parallel_tasks` / `time.parallel_tasks` | 独立工作并发预算 |

`accuracy.field_cap` 是兼容旧输入的默认场上限。下面的可选参数未设置时均继承它；显式设置后只控制对应对象：

| 参数 | 消费者 |
|---|---|
| `external_cap` | 静态外势的 QTT 构造 |
| `kernel_cap` | 相互作用核 QTT；循环 MPO 的内部 bond 为核 bond 的两倍 |
| `density_cap` | 输入/输出密度拟合及混合密度生成 |
| `hartree_cap` | 核 MPO 作用于密度后的 Hartree 势 |
| `effective_potential_cap` | 有效势 QTT/对角 MPO，以及固定势求解的势表示 |
| `xc_energy_cap` | XC 能量密度的 QTT 构造 |
| `drive_cap` | 计算外场能量时的外场 QTT；传播相位仍由 `time.phase_cap` 控制 |

例如保持 `field_cap=128`，只设置 `external_cap=48`、`kernel_cap=16`，其余场仍继承 128。这是分开预算的配置示例，不代表这些值已通过完整 SCF 精度验收。`plan` 和 `config.json` 的 `resolved_field_caps` 列出全部实际生效的上限。审计的 `rank_multiplier` 同时放大继承值和显式覆盖值。

局域势模型在已有外势 rank 满足有效势上限时直接复用外势核心；显式要求更小的有效势上限时才重新拟合。XC 势目前作为密度的标量函数进入总有效势，没有单独的 XC 势 MPO。

`[run].checkpoint` 相对于输入文件目录解析。它导入轨道，SCF 历史从第一轮重新开始；同目录有效的 `cross-state.jls` 可用于 pivot 热启动。细化目标网格时使用 `[refinement]`。输出路径相对于启动目录；命令行 `NEW_OUTPUT` 优先于输入中的 `run.output`。

每次运行保存 `input.toml`、`config.json`、`status.json`、迭代日志和数据检查点。`workflow.json` 记录配置运行的时间范围；`source/` 保存完整包的 `src/`、依赖声明及 `run.jl`。可以将 `source/` 作为 Julia 项目启动，并以保存的输入进行新的计算；重启所引用的外部检查点须继续可用。

SCF 的 `scf.jsonl` 每行是一轮墙钟时间，初始化、检查点写入及末尾审计不都包含在这一行内。并行子任务时间可能重叠，不能相加后当作整轮墙钟时间。轨道核存储、累计分配和峰值内存是不同量。

轨道上限 `accuracy.orbital_cap` 与实际 bond dimension 分开记录：`inner_output_max_rank` 是 L7 后、保存压缩前的最大 bond；`orbital_rank_before_pack` 是本轮首次保存压缩的输入最大 bond；`stored_max_rank` 是最终输出轨道的最大 bond（包括 strict 模式输出 Ritz 后的再次保存）。兼容字段 `lrrap.jsonl.max_rank` 仍表示内层输出，`scf.jsonl.max_rank` 仍表示最终保存轨道。关闭 TCI 后置 SVD 不会关闭由 `storage_tolerance` 控制的输出压缩。

能量停止与物理接受分别保存在 `status.json` 的 `converged` 与 `audit.passed`。`criterion=energy` 可以在物理审计未通过时停止；首轮没有相邻能量差。命令行同时显示停止结果和审计结果，不合并成含义不明的“已接受”。
