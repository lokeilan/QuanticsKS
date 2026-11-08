# 一维 QuanticsKS 算法

3D extension: [models, parameters, numerical conventions, and limitations](three-dimensional.md). `src/models3d.jl` defines geometry and models; `src/three_dimensional.jl` supplies Cartesian operators and adapters to the shared SCF workflow.


## 网格与 Hamiltonian

计算采用原子单位和一维周期均匀网格。`R` 个二进制站点表示 $N=2^R$ 个网格点，间距为 $h=L/N$。QTT 核按低有效位到高有效位排列，存储轨道振幅 $u_{ij}=\sqrt{h}\,\psi_i(x_j)$。因此轨道使用普通欧氏内积，物理密度为 $n_j=\sum_i f_i |u_{ij}|^2/h$，其中 $f_i$ 为占据数。

动能 $T$ 采用周期二阶中心差分（FD2）。

$$
(Tu)_j=\frac{(u_j-u_{j-1})+(u_j-u_{j+1})}{2mh^2}.
$$

Hamiltonian 由动能和对角局域势组成。对软库仑模型，局域势包括外势、Hartree 势和 XC 势。Hartree 势通过周期相互作用核的循环 MPO 作用于密度，再乘以积分权重 $h$。势的 QTT 核供直接投影使用，残差和 Green 右端可共享原始局域势的采样表达式。 [场与 Hamiltonian 的构造](../src/hamiltonian.jl)

## SCF 外层循环

1. 构造静态外势、相互作用核、周期移位和所需的 XC 表，准备初始轨道和 Green MPO。
2. 从当前轨道拟合密度，并把积分电荷归一到占据数之和。用线性或 Anderson 混合得到本轮输入密度。混合历史保存原 QTT 核的线性表达式，在构造有效势前生成混合密度的独立 QTT 核。
3. 从混合输入密度构造 $H$，在本轮的 `inner_steps` 次 LRRAP 更新中固定该算符，并复用局域势的采样缓存。
4. 将更新后的轨道按保存精度舍入。保存压缩检查实际轨道的 Gram 矩阵，必要时收紧舍入容差。随后从保存轨道重算密度。`strict` 模式还用输出密度对应的 $H$ 做直接投影 Ritz，再保存旋转后的轨道并重算密度。
5. 计算输出总能量、相邻轮的能量变化，以及输出密度与本轮混合输入密度的相对差。使用所选条件判断是否停止。 [SCF 循环](../src/scf.jl) · [输出轨道 Ritz](../src/scf.jl)

## LRRAP 内层更新

| 日志阶段 | 操作 |
|---|---|
| L1 | 直接收缩轨道的 Rayleigh 值，并用 TCI 拟合完整残差 $Hu_i-\varepsilon_i u_i$。 |
| L2 | 从轮换选取的占据轨道建立共同的规范切空间框架。 |
| L3 | 将轨道、Green 预条件残差及可用的历史方向投影到同一坐标系。Green 作用直接进入投影收缩。 |
| L4 | 在共同坐标中做 SVD，舍去小奇异值方向。沿切空间方向直接收缩构造约化 $H$，并保留轨道与方向之间的耦合。 |
| L5 | 在小型系数问题中求解轨道更新，施加占据轨道的正交约束。 |
| L6 | 把修正重构为 QTT 轨道。安排 Green 修正时，历史方向在本步清零。 |
| L6G | 按配置间隔或 L1 最大残差阈值触发。用 L1 的 $\varepsilon_i$ 和 L6 重构的轨道拟合 $(\mu+\varepsilon_i-V)u_i$，施加 $(T+\mu I)^{-1}$，再归一化。 |
| L7 | 检查并恢复轨道正交性。触发 Green 修正时，还按投影矩阵的非对角量决定是否执行 Ritz 旋转。 |

L4 的直接投影使用已经满足规范约束的切空间坐标。L7 在工作容差为正时，可先拟合每条轨道的 $Hu_i$ 再构造投影矩阵。工作容差为零时，直接收缩 $\langle u_i,Hu_j\rangle$。正交变换和 Ritz 旋转先在小矩阵中合并，最后重构 QTT 轨道并复查 Gram 矩阵。 [LRRAP 阶段](../src/numerics/lrrap.jl) · [正交化与 Ritz](../src/numerics/lrrap.jl)

## 停止与审计

`energy` 模式以相邻两轮总能量差作为停止条件。`strict` 模式同时要求密度差、保存轨道的 Gram 缺陷和独立审计满足阈值。`status.json` 分别记录停止结果 `converged` 和审计结果 `audit.passed`。

独立审计从最终轨道重算密度与 $H$，检查逐轨道的相对本征残差 $\|Hu_i-\varepsilon_i u_i\|_2/\|u_i\|_2$、Gram 缺陷、电子数、密度自洽差和总能量重算差。它使用独立的精度设置。 [审计实现](../src/scf.jl) · [配置和输出字段](configuration.md)

### 固定势残差诊断

固定输出 $H$ 时，残差可分成占据子空间内的耦合和子空间外的分量。精确 Ritz 旋转可消除前者。若旋转后仍有较大残差，需要改变占据子空间。

保存舍入产生的小幅高频误差会被 FD2 动能放大，Gram 缺陷很小的轨道也可能有较大本征残差。可以用同一动能谱做低通对照，给动能为 $\lambda$ 的频率分量施加 $1/[1+(\lambda/\Lambda)^2]$ 的振幅权重，$\Lambda$ 为增益降到一半时的动能。滤波作用于轨道，随后重新正交化，并用原 $H$ 测量残差。该对照用于分辨高频误差与尚未收敛的低频分量。

## QTT 构建与并行

TCI 通过 prrLU 自适应选择采样点并构建张量，默认保留该结果。`post_rounding_factor=0` 表示关闭后置 SVD；设置正值时按 `tolerance * post_rounding_factor` 显式舍入。切空间基由 ITensorMPS 的无截断规范化建立。默认 MPO 作用使用库的 zip-up，由库同时规范化 MPO 和轨道并完成最终截断；外层不再追加 SVD。项目的振幅容差 `tol` 对应 ITensor 的平方权重阈值：局部 `cutoff=(tol/R)^2`，最终 `cutoff=tol^2`，秩上限仍可能主导误差。零容差且秩足够时直接执行库的无截断乘积。存储阶段按 `storage_tolerance` 控制最终轨道精度。TCI 的 pivot 误差与全局误差、物理残差分别测量。 [TCI 构建](../src/numerics/tci.jl) · [MPO 作用](../src/numerics/tensor.jl)

独立轨道拟合、投影、MPO 作用和 Gram 矩阵计算可以并行。`parallel_tasks=0` 使用可用的 Julia 工作线程，`1` 为串行。`inner` 调用 TCI 的精确收缩，外层任务调度控制并行预算。 [线程调度](../src/numerics/parallel.jl) · [内积实现](../src/numerics/tensor.jl)

## 固定势谱与实时传播

`fixed_spectrum` 在给定局域势下求本征谱，可选 Block Green、残差子空间更新或 LRRAP。它按占据带能量变化停止，并单独记录本征残差。 [固定势谱求解](../src/spectrum.jl)

`propagate` 依次执行势能半步、FD2 Cayley 动能整步和势能半步。两个势能半步各自从输入轨道更新密度，外场在时间中点求值。传播记录轨道范数和 Gram 矩阵的漂移。 [实时传播](../src/dynamics.jl)

## 本地低通实验接口

工作区的独立低通 MPO 复用周期 Green 核，最大内部秩为 8，存储量随网格位数线性增长。以下示例使用 R16 对照的截止值，单位为 Ha。

```julia
filter = lowpass_mpo(state.problem; cutoff = 1000.0)
filtered = apply_lowpass(filter, state; tol = 1e-12, cap = 64, parallel_tasks = 0)
```

`apply_lowpass` 保留滤波造成的振幅衰减。用作求解初态前，应恢复归一和正交，并以原 Hamiltonian 检查最终 QTT 轨道的残差。
