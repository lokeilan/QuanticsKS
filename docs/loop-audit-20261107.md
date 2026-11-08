# 循环与软件包复用核查（2026-11-07）

后续状态：MPO 矩阵元、规范化和默认 zip-up 已在 2026-11-08 迁到 ITensorMPS，见[迁移与三维核对](itensormps-3d-20261108.md)。下文保留本轮核查时的判断和比较数据。

仍有可以由库接管的手写运算。上一轮整理遗漏了张量积分、普通系数乘积和部分重复压缩；本次已修正。更大的重复实现集中在 MPO 作用、规范化和新增三维 Green 求解中，需要按完整的计算接口迁移。

判断依据是循环承担的计算职责。把 `for` 改写为 `map`，或者把相同代码藏进 wrapper，并不算复用了库。

本次检查 `src` 中的显式循环、推导式及其调用关系，并对照本机锁定的 TensorCrossInterpolation 0.9.19 源码。外部替代接口另查官方文档。工作期间新增的 `models3d.jl` 和 `three_dimensional.jl` 也做了只读评估；下面的运行验证针对原有一维流程和共用数值层，不代表三维实现已经验证。

## 已完成的替换

| 位置 | 原实现 | 现在的实现与职责 |
|---|---|---|
| `hamiltonian.jl` 的密度归一化、XC 积分；`numerics/lrrap.jl` 的混合密度积分 | 构造全 1 QTT，再求内积 | 直接 `sum(Q.tensortrain(rho))` 或相应能量密度求和；体积元仍在物理层乘一次 |
| `numerics/sampling.jl` 的 `LinearSamples` | `weighted_value` 手写加权循环 | `transpose(weights) * values`；复系数不取共轭 |
| `numerics/tensor.jl` 的 `expression_gram` | 逐项累加系数二次型 | `W' * K * W`；小范数和近线性相关仍转入振幅空间 QR，避免大数相减求残差范数 |
| `numerics/lrrap.jl` 的 Ritz 系数和系数解归一化 | 按列切片、除以列范数 | 对已有列范数做广播，保留有限性与正范数判断 |
| `numerics/tensor.jl` 的 `circulant` | 遍历左右键指标逐元素复制核心块 | 数组视图加广播；外层二进制借位状态仍表示循环卷积的定义 |
| `numerics/tensor.jl` 的 `apply_mpo` 非默认分支 | 库按临时秩压缩/拟合，随后再 SVD 压缩 | 将最终 `cap` 直接交给 `TCI.contract`，返回其结果；`:naive` 的压缩和 `:TCI` 的构造各自负责秩控制 |

TCI 的张量求和沿核心收缩，不会展开整个网格。其 `contract_naive` 已调用 `compress!`，`contract_TCI` 已将容差和秩上限交给交叉插值；外面没有必要统一再接一次 SVD。[TCI 官方接口](https://tensor4all.org/TensorCrossInterpolation.jl/dev/documentation/)

本次没有改变默认的本地 zip-up 算法，也没有引入新依赖。

## 有现成实现，仍应继续迁移的部分

| 本地实现 | 可用接口 | 评估 |
|---|---|---|
| `inner_mpo` 的逐核心三层收缩 | ITensorMPS `inner(y, A, x)` | 已有直接计算矩阵元的库函数。属于明确的替换候选；应与 QTT/MPS 数据接口一起迁移 |
| `right_canonical`、`canonical_frame` 的规范变换 | ITensorMPS `orthogonalize` / `orthogonalize!` | 库支持无截断规范化。frame 所需的左右核心、中心位置和投影布局仍由消费者确定 |
| 默认 `apply_mpo` 的 zip-up | TCI `contract(...; algorithm=:zipup)`；ITensorMPS `apply` / `contract` | 现成实现确实存在。当前左正交局部截断和锁定库的截断过程不同；应选定一种误差与秩策略后统一采用，不能据此宣称本地实现永久必要 |
| `PointwiseSamples` 等逐点组合后重构 | AlternatingCrossInterpolation `elementwise` | 可以接管 TT 逐点运算，但会改变拟合算法；当前 pivot 重启、共享采样、缓存上限和统计需要重新对接 |
| `models3d.jl` 的 `grid_indices` / `grid_index`，以及一维 `bits` / `indexof` | QuanticsGrids 坐标与 quantics 索引转换 | 三维映射已有通用库。迁移需统一当前零基索引、最低有效位在前、交错轴顺序和单元中心位置；短小的一维位操作可作为同一映射接口的适配部分 |
| `three_dimensional.jl` 的 `CartesianGreen` 手写 CG | KrylovKit `linsolve` / `CG`，配合 VectorInterface | 通用 CG 递推应优先交给库。QTT 加法后的截断、真实残差复核和重启策略需明确放在适配层，不能把近似线性运算当作精确向量运算 |

ITensorMPS 的矩阵元、无截断规范化和 MPO 作用接口见[官方文档](https://docs.itensor.org/ITensorMPS/stable/MPSandMPO.html)。如果采用这个后端，宜在数值层建立统一的数据边界，一次接管相关运算，避免每个小函数反复转换并增加 wrapper。

其他候选的接口来源分别是 [AlternatingCrossInterpolation](https://github.com/tensor4all/AlternatingCrossInterpolation.jl)、[QuanticsGrids](https://tensor4all.org/QuanticsGrids.jl/dev/)、[KrylovKit 线性求解](https://jutho.github.io/KrylovKit.jl/stable/man/linear/)和 [VectorInterface](https://quantumkithub.github.io/VectorInterface.jl/stable/man/interface/)。这些候选尚未加入项目依赖，也没有在本次核查中完成生产规模性能评估。

### zip-up 的实际比较

使用复数随机 QTT、周期离散 Green MPO 和独立 dense 逆算子，在每个比较中给两种实现相同输入、临时秩上限及最终舍入设置。库内局部容差设为 `tol/R`，与本地局部预算对齐。

| 网格与秩 | 本地结果相对 L2 误差 | 库结果相对 L2 误差 | 两个结果的相对 L2 差 |
|---|---:|---:|---:|
| R=8，cap=16，tol=0 | 6.56e-15 | 7.52e-15 | 9.06e-15 |
| R=8，cap=4，tol=0 | 9.38e-3 | 1.00e-2 | 3.49e-3 |

这里的相对误差均以 dense 目标的范数归一化。秩充足时结果接近浮点精度；秩受限时确实出现截断差异。这支持“可以迁移，但不是保持当前截断结果的直接替换”，没有证明哪一种在生产问题上更快或更准确。不同表格行使用不同随机输入，不用于跨秩比较算法优劣。

## 应保留或另行处理的循环

| 范围 | 判断 |
|---|---|
| `scf.jl`、`spectrum.jl`、`dynamics.jl` 的外层迭代 | 下一轮依赖上一轮密度、子空间或状态，循环就是算法流程；不能仅为消除 `for` 改写 |
| `numerics/lrrap.jl` 的系数扫描 | 使用刚更新的系数，具有顺序依赖；内部特征分解与矩阵求解已经调用 LinearAlgebra |
| `gram`、`h_gram` 的轨道对遍历 | 每个条目调用 TT 收缩，Hermitian 对称避免重复计算；`inner_mpo` 本身仍属于上面的库迁移候选 |
| `expression_qr` 与 `compensated_sum` | 共同 QR 基和补偿振幅求和用于处理相消；普通二次型已经移交矩阵乘法。直接用大范数相减不能替代这项职责 |
| `sampling.jl` 的有界缓存、缺失样本分组 | TCI 的 `TTCache` 已接管可复用的端点环境；它不等同于当前有容量限制和淘汰统计的采样缓存。后者可评估通用缓存库，但不应换成无界字典 |
| `svd_safety.jl` 的恢复扫描 | 正常路径先调用 `TCI.compress!`；本地扫描仅用于数值失败后的诊断恢复，锁定库没有分解后端回调。尾平方和的逐尾重算仍可进一步改为累计求和，需保留严格阈值和零容差约定 |
| `physics.jl`、MPO 构造及三维边界处理 | 物理核、借位/进位、轴与边界条件是模型定义；Bessel、zeta、积分及局部矩阵收缩已经交给相应库。通用 CG 与索引转换另列为迁移候选 |
| `PeriodicRefinement` 的局部插值 | 当前为周期局部 Lagrange 多项式。可另评估插值包，但普通样条并非同一算子；目前没有接入经过等价核对的包接口 |
| `configuration.jl`、`states.jl`、`observables.jl`、`tci.jl`、`parallel.jl` 的遍历 | 配置、检查点、输出、pivot/日志归并和任务调度；这些循环不构成重新实现 TCI 数值算法 |

## 验证范围与记录

复数加权和、TT 直接求和、表达式 Gram 与独立 dense 收缩一致；近乎抵消的表达式使用高精度核心收缩作为参照，仍能解析非零的小残差。MPO 分支保持预期的周期位移作用和秩上限，输入核心未被修改。小网格一维场、Hamiltonian 和总能量的独立 dense 对照未发现本次修改引入的不一致。

还检查了 pivot 重启、独立场秩设置及共用数值层的串行/多线程行为。最后修改的 Ritz 列归一化在固定势、输出 Ritz、网格细化和传播流程中保持预期行为。固定工作量的一维 SCF 串行/并行能量差为 0 Ha；末态审计残差为 5.43e-3 Ha，未达到物理接受条件。

实际执行记录保存在 `outputs/loop-audit-20261107/`：`focused.log`、`algebra.log`、`parallel.log`、`ritz-and-workflows.log`；`loop-inventory.json` 保存循环位置，`zipup-comparison.json` 保存比较数据。原始日志保留 Julia 自带的输出格式，本文只说明行为和结果。

没有运行一维 R16 生产任务，也没有将这些局部验证解释为 SCF 物理收敛或三维模型接受。
