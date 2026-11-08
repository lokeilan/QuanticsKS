# ITensorMPS 迁移与三维核对（2026-11-08）

通用 MPO 矩阵元、规范化和默认 zip-up 已迁到 ITensorMPS。本次还修正了三维 Green 接受判断中的残差再截断，以及局域势复用时遗漏独立秩上限的问题。活动源码保持现有文件结构，检查点继续保存 QTT 核。

## 库调用与计算职责

| 位置 | 当前实现 |
|---|---|
| `numerics/tensor.jl: inner_mpo` | `ITensorMPS.inner(bra, operator, ket)`，不先构造近似的 `operator*ket` |
| `numerics/tensor.jl: right_canonical` | `ITensorMPS.orthogonalize!`，不设置截断容差或秩上限 |
| `numerics/hamiltonian.jl: canonical_frame` | 同一个 MPS 先移到左端中心、保存右规范核心，再移到右端中心、保存左规范核心 |
| `numerics/tensor.jl: apply_mpo` | 默认使用 `ITensorMPS.apply(...; alg="zipup")`；`:naive` 也使用 ITensorMPS；显式 `:TCI` 继续使用 TCI 构造 |
| 一维和三维 `h_gram` | 每次投影只转换一次轨道集合和各 MPO，矩阵元素直接使用库收缩；一维串行与并行合并为同一实现 |

这些接口见 [ITensorMPS 官方文档](https://docs.itensor.org/ITensorMPS/stable/MPSandMPO.html)。MPS/MPO 与 TensorTrain 的转换使用 TCI 自带的扩展。TCI 0.9.19 的反向转换在单站点和双站点上有缺口，因此仅这两种情况直接取出端点数组；较长张量仍由库转换。

`apply_mpo` 的 `tol` 仍表示振幅误差尺度。ITensor 的 `cutoff` 是被丢弃的平方权重比例，故局部使用 `(tol/R)^2`，最终使用 `tol^2`；`R` 在此是张量站点总数。临时秩为 `max(2cap,16)`，最终秩为 `cap`。库自行规范化两个输入并完成最终截断，外层不再追加 SVD。这个截断流程与旧实现不同，限秩输出不要求逐核心或逐位一致；误差通过独立物理量核对。

零容差且最终秩足以容纳乘积时，调用库的无截断乘积，避免先分解再恢复同一结果。旧的 `svd_tail_rank` 和 `input_right_canonical` 标志已删除，调用方不再猜测转换后的库对象是否保留了规范元数据。

ITensor 索引 ID 会使用随机数。相关计算在保存和恢复随机流的作用域内执行，避免影响 TCI 采样；复数矩阵元、输入不被修改、端点转换、等价规范和随机流保持均已核对。

TCI 拟合、普通 TT 内积/相加、压缩和有界采样继续使用原来的库接口。切空间投影与模型 MPO 构造保留在项目内。三维截断 CG 尚未迁到 KrylovKit，坐标编码也尚未迁到 QuanticsGrids；本次没有把这两项写成已经完成。

## 三维修正

### Green 的接受判断

原来的“真实残差”通过 `combine(rhs, action(result))` 构造，其中 `action` 和 `combine` 都会按工作秩截断。因此，被截断掉的误差可能同时从接受指标中消失；原判断还额外允许两倍配置容差。

现在对最终存储结果构造

\[
r=b-(T_{\rm FD2}+\mu I)\psi.
\]

六个开边界移位使用无截断乘积，残差保持为线性表达式。范数使用核心收缩，发生相消时用共同 QR 基上的振幅求和。该接受路径不按 CG 工作秩截断，要求相对残差不超过配置的 `green_relative_tolerance`。需要重启搜索时，才将残差压缩为工作向量。

回归将一个解压到秩 1，再与 dense 离散算子比较残差；新指标能看到这部分存储误差。无法满足最终存储秩下误差预算的求解会报错。

### 独立势场预算

`LocalPotential3D` 原来无条件复用外势 QTT，即使 `effective_potential_cap` 比外势秩更小。现在仅在外势符合有效势预算时复用，否则按有效势自身的预算构造；行为与一维入口一致。

## 物理与离散约定核对

| 项目 | 核对方式与结论 |
|---|---|
| 轴与边界 | 非等边盒、非单位质量、复轨道下，FD2 作用与独立 dense 矩阵一致；轴端点不回绕、不串入其他轴 |
| Green | 使用相同 FD2 与正位移 `mu`；小网格离散逆作用的相对残差约 `8.48e-10`，对应设置为 `1e-9` |
| Hartree | 保留 TCI 完整收缩环境采样，避免对原始借位 MPO 前缀直接截断；有限自作用、带符号 Toeplitz 位移和 dense 卷积一致 |
| 核外势与积分 | 轨道为 `sqrt(dV)*psi`；密度除以 `dV`，空间积分与 Hartree 卷积各在对应物理步骤乘一次 `dV` |
| XC | 一维软库仑 LSDA 与三维 PZ81 分开；三维系数与 Libxc 的原始非极化 PZ81 参数表一致，势与能量密度的导数相符 |
| 状态与入口 | 多轨道执行、细化、检查点身份和 3D 输入选择正常；未实现的 3D 传播、固定谱和低通接口仍明确拒绝 |

PZ81 参数依据是 [Libxc 源码](https://sources.debian.org/src/libxc/5.2.3-3.1/src/lda_c_pz.c/)，采用原始 `PZ`，没有暗中改成 `PZ_MOD`。库迁移没有改变空间边界、核势定义、占据或 XC 模型。

## 离散氦模型的独立 dense 对照

`scripts/validate_3d.jl` 运行盒长 8 bohr、`R=3` 每轴的孤立氦模型，即 `8^3` 网格。轨道、密度和场通过 QTT 求解；dense 参照另外组装 FD2、核势和库仑矩阵并完成 SCF。两者共享标量 Coulomb/XC 定义，这些标量公式另用独立积分和导数核对。

| 测量 | 结果 |
|---|---:|
| QTT 总能量 | -1.974332226458436 Ha |
| dense 总能量 | -1.9743322264584557 Ha |
| 总能量绝对差 | 1.98e-14 Ha |
| 密度相对 L2 差 | 1.55e-8 |
| 独立审计轨道残差 | 1.37e-7 Ha |
| 独立审计密度残差 | 5.03e-7 |
| 电子数 | 1.9999999999999996 |
| Gram 缺陷 | 2.22e-16 |
| 停止条件 | strict |
| 本次配置的物理接受 | 满足 |

输入、源码快照、检查点、机器可读测量和报告保存在 `outputs/itensormps-3d-audit-20261108/helium-r3/`。报告生成器同时更新：分别记录停止判据、停止原因和物理接受，并记录实际 ITensor 包版本。

一维方面，复数张量代数、近相消表达式、独立 dense 场/Hamiltonian/能量、固定势、Ritz、传播与串并行行为保持预期。短程一维 SCF 的能量一致不等于物理收敛，其原有独立审计仍报告未接受。

这些结果验证当前小网格离散实现的一致性，不代表连续空间精度、盒长收敛、论文能量复现或 H12/生产规模性能。远场 Coulomb 体素平均仍采用固定 Gaussian 求积，其误差也不等同于 TCI 拟合容差。

## 依赖与执行记录

当前锁定 ITensorMPS 0.3.45、ITensors 0.9.32，TCI 保持 0.9.19。兼容解析调整了间接依赖 DocStringExtensions、VectorInterface、Parsers 和 Strided，原有直接数值依赖的版本保持不变。没有修改全局 Julia 环境配置。

主要记录在 `outputs/itensormps-3d-audit-20261108/`：`algebra-recheck.log`、`workflows.log`、`three-dimensional-final.log`、`compression-and-driven.log`。最初的依赖冲突与双站点转换失败日志也保留；对应问题已修正。

## 运行开销

ITensor 的对象转换和通用索引操作有实际开销，迁移不意味着所有核都会更快。以下是合成 R16、秩上限 32 的预热调用中位时间，BLAS 使用单线程。MPO 是离散 Green 算子；Hamiltonian 投影使用四条合成轨道。计时包含表示转换与分配，不包含进程启动和首次编译。

| 操作 | 迁移前 | 当前 | 迁移前 / 当前累计分配 |
|---|---:|---:|---:|
| 单个 MPO 矩阵元 | 1.41 ms | 2.01 ms | 4.20 / 6.75 MiB |
| 左右规范框架 | 1.97 ms | 3.28 ms | 4.92 / 10.26 MiB |
| MPO 作用 | 34.69 ms | 22.37 ms | 35.90 / 35.88 MiB |
| Hamiltonian 投影矩阵 | 31.31 ms | 42.24 ms | 74.63 / 115.47 MiB |

投影矩阵与迁移前结果的 Frobenius 差为 `6.03e-15`。当前 MPO 作用在这个合成例子上更快，矩阵元、规范化和投影仍有额外开销；更小秩的例子中通用对象开销更明显。累计分配量不是进程峰值内存。这些局部结果不能推断完整 SCF 的加速比。

完整数据在 `kernel-costs-final.json`，初步计时保留为 `kernel-costs.json`。复现脚本为 `.tooling/benchmark_itensormps_20261108.jl NEW_OUTPUT.json`，它读取本次开始前的源码副本作为对照。尚未运行生产规模性能比较。
