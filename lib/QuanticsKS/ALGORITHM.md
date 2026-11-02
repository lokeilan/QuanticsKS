# QuanticsKS 算法说明

本文说明当前一维周期 Kohn–Sham 程序如何计算。术语采用数值线性代数、张量计算和电子结构文献中的名称；公式与步骤以随包源码为准。文献说明方法来源，本项目的计算时间和误差来自保存的原机记录。

## 1. 程序求解的三个问题

| 计算任务 | 输入 | 迭代过程中改变什么 | 输出 |
|---|---|---|---|
| Kohn–Sham 自洽场迭代（SCF） | 物理模型、占据数、初始轨道 | 轨道、电子密度、Hartree 势、交换关联势 | 按所选 SCF 判据停止的轨道与总能量 |
| 固定 Hamiltonian 本征值问题 | 已保存的轨道、固定有效势 | 轨道和近似本征值 | 固定有效势下的 Ritz 轨道和 Ritz 值 |
| 实时含时 Kohn–Sham 传播（RT-TDKS） | 初始轨道、时间步长、外场 | 轨道随时间变化，密度反馈到有效势 | 时间序列、密度、能量和其他观测量 |

第一项使用低秩黎曼交替投影更新轨道。第二项默认使用块 Green 迭代与 Rayleigh–Ritz 投影。第三项使用 Strang 算符分裂与 Cayley 动能传播。

当前交付的 R16 数值记录属于第二项：固定 Kohn–Sham Hamiltonian，求解子空间包含 50 个双占据轨道和 5 个辅助未占据轨道。该计算未将更新后的密度重新送入 SCF。

## 2. 术语与符号

后文的符号流程图均用纯文本表示，可直接复制到普通 Markdown 阅读器，无需 Mermaid 或图片插件。

| 术语 | 英文或缩写 | 本文含义 |
|---|---|---|
| Kohn–Sham 轨道 | KS orbital | 构成电子密度的单粒子辅助轨道 |
| 占据数 | occupation number，\(f_i\) | 每个空间轨道承载的电子数 |
| 未占据轨道／虚轨道 | unoccupied / virtual orbital | 占据数为零的轨道；可用于扩大迭代子空间 |
| 自洽场迭代 | self-consistent field，SCF | 在轨道、密度和有效势之间重复更新 |
| 量化张量列 | quantics tensor train，QTT | 将网格索引拆为二进制指标后得到的张量列表示 |
| 张量列／矩阵乘积态 | tensor train，TT / matrix product state，MPS | 由顺序连接的低阶核心张量表示一个高阶张量 |
| 键维数／TT 秩 | bond dimension / TT rank | 相邻核心之间的内部指标维数 |
| 张量交叉插值 | tensor cross interpolation，TCI | 根据选定张量元素构造 TT 近似 |
| 部分秩揭示 LU 分解 | partially rank-revealing LU，prrLU | 当前 TCI 库使用的秩揭示分解 |
| 矩阵乘积算符 | matrix product operator，MPO | 算符的张量列表示 |
| 低秩黎曼交替投影 | low-rank Riemannian alternating projection，LRRAP | 在交替选择的 TT 切空间中构造多轨道更新 |
| 预条件器 | preconditioner | 改变修正方向的谱分布，以改善迭代行为的算符 |
| Rayleigh–Ritz 投影 | Rayleigh–Ritz projection | 在有限维试探子空间中求解投影本征值问题 |
| Ritz 值／Ritz 向量 | Ritz value / Ritz vector | 投影本征值问题给出的近似本征值与轨道 |
| 重叠矩阵／Gram 矩阵 | overlap / Gram matrix，\(S\) | \(S_{ij}=\langle u_i,u_j\rangle\) |
| TT 截断／舍入 | TT truncation / TT rounding | 在精度和最大秩限制下压缩 TT 核心 |
| 网格延拓 | grid prolongation | 从粗网格表示构造细网格初始轨道 |
| Strang 算符分裂 | Strang splitting | 对称的半步—整步—半步时间推进 |
| Cayley 传播子 | Cayley propagator | 用有理函数近似动能时间演化算符 |

QTT 与 TCI 的名称分别依据 [Khoromskij 的 QTT 工作](https://www.mis.mpg.de/de/publications/preprint-repository/article/2009/issue-55)和 [Núñez Fernández 等的 TCI 论文](https://arxiv.org/abs/2407.02454)。LRRAP 名称依据 [Rakhuba、Novikov 与 Oseledets](https://arxiv.org/abs/1811.11049)。中文译名首次出现时保留英文，避免缩写歧义。

## 3. 网格、轨道归一化与电子密度

程序采用一维周期网格：

\[
N=2^R,\qquad h=L/N,\qquad x_j=jh,\quad j=0,\ldots,N-1.
\]

晶胞端点 \(L\) 与 0 是同一点，因此不重复采样。`R` 是网格指数；核间距由 `distance` 单独设置。

R16 一维 100H 算例使用 \(L=400\) bohr、\(N=65536\)、100 个 H 核和 100 个电子。restricted 自旋非极化模型使用 50 个空间轨道，每个轨道占据数 \(f_i=2\)。

程序保存的是带积分权重的离散振幅：

\[
u_{i,j}=\sqrt h\,\psi_i(x_j).
\]

因此，连续归一化的离散对应为 \(\sum_j|u_{i,j}|^2=1\)。电子密度和电子数为：

\[
n_j=\frac1h\sum_{i=1}^{N_{\rm occ}}f_i|u_{i,j}|^2,
\qquad N_e=h\sum_jn_j.
\]

固定 Hamiltonian 算例中的 5 个辅助未占据轨道不进入这个求和。求解更多轨道不会改变电子数。

## 4. QTT 如何存储轨道

把网格索引写为二进制：

\[
j=\sum_{k=1}^{R}b_k2^{k-1},\qquad b_k\in\{0,1\}.
\]

一个长度为 \(2^R\) 的轨道向量成为具有 R 个二值指标的张量。QTT 将它写为核心矩阵的连乘：

\[
u_j=G_1[b_1]G_2[b_2]\cdots G_R[b_R].
\]

核心 \(G_k\) 的尺寸是 \(r_{k-1}\times2\times r_k\)，边界秩为 \(r_0=r_R=1\)。保存的参数数目为 \(\sum_k2r_{k-1}r_k\)，而直接保存网格振幅需要 \(2^R\) 个参数。只有实际 TT 秩保持适中时，QTT 才有存储优势。

程序内部核心按最低有效位到最高有效位排列，即 LSB→MSB。bond profile 为方便观察采用反向的 MSB→LSB 顺序。图中第 k 个 bond 的代数秩上界为：

\[
r_k\le 2^{\min(k,R-k)}.
\]

`orbital_cap` 是保存轨道的最大允许 TT 秩，`work_cap` 控制中间张量，`field_cap` 控制密度和势。它们作用于不同对象，不会自动同步。

本次最终轨道的最大实际键维数为 45，轨道秩上限为 48。逐 bond 中位数是对 55 条轨道在同一 bond 上的键维数取中位数；它不代表某一条特定轨道。

## 5. TCI、prrLU、QR 与 SVD 分别做什么

TCI 接受一个“按索引返回函数值”的回调。例如，构造势与轨道的乘积时，只需提供 \(j\mapsto V_j u_j\)。TCI 自适应选择需要查询的张量元素，再构造其 TT 近似；本程序调用 `TensorCrossInterpolation.crossinterpolate2`。

该库的算法采用 prrLU。它属于交叉插值的秩揭示步骤，参见 [TCI 原始论文](https://arxiv.org/abs/2407.02454)。

当前程序另外保留两类线性代数操作：

- **QR 分解**：构造正交规范形式，为 TT 截断和切空间计算提供稳定的表示。
- **奇异值分解（SVD）**：在 TT 舍入中按奇异值截断，也用于压缩共同切空间坐标及处理系数优化中的约束。

一次 TCI 调用结束后，程序还会以 `TCI tolerance × post_rounding_factor` 做 TT 舍入。因此最终键维数同时受到函数本身、TCI 容差、舍入容差、秩上限和轨道基选择影响。

把已知稠密轨道作为缓存函数交给 TCI，只测量该给定轨道的可压缩性。原生 QTT 迭代则不断生成算符作用结果和新轨道，二者的键维数不必相同。比较时应使用相同 Hamiltonian、相同轨道或明确的子空间对齐、相同精度和相同秩上限。

## 6. Kohn–Sham Hamiltonian 的构造

对当前局域势模型：

\[
H[n]=T+V_{\rm ext}+V_{\rm H}[n]+V_{\rm xc}[n].
\]

动能采用二阶中心有限差分（FD2）：

\[
(Tu)_j=\frac{2u_j-u_{j-1}-u_{j+1}}{2mh^2},
\]

左右索引按周期边界取模。平移算符和局域势分别具有 MPO 表示。

一维氢链使用周期软库仑模型。外势来自各原子核势之和；Hartree 势为密度与同一相互作用核的周期离散卷积：

\[
(V_{\rm H})_j=h\sum_\ell w(x_j-x_\ell)n_\ell.
\]

该卷积通过循环矩阵的 MPO 作用计算。周期软库仑核采用代码中固定的对称周期像求和与尾项展开，不等同于三维库仑势。

交换关联势使用现有一维局域自旋密度近似（LSDA）的自旋非极化分支。实现先构造交换关联能量密度及导数的插值表；表外直接计算同一公式。当前 LSDA 参数化只适用于软化长度、耦合常数和电子质量均为 1 的设置。

在固定 Hamiltonian 本征值计算中，`fixed-potential.f64` 已经给出有效势 \(V\)。迭代期间不会重新计算 Hartree 或交换关联势。默认块 Green 更新、Rayleigh–Ritz 矩阵和残差诊断均使用原始势样本。

## 7. SCF 外层迭代

```text
物理模型、网格、占据数、QTT 初始轨道 U
                    |
                    v
          n_in = sum_i f_i |u_i|^2 / h
                    |
                    v
       线性密度混合 / Anderson 密度混合
                    |
                    v
        H[n] = T + V_ext + V_H[n] + V_xc[n]
                    |
                    v
       LRRAP 轨道更新（见下方切空间流程）
                    |
                    v
        TT 截断 -> 新轨道 U -> 输出密度 n_out
                    |
                    v
     E_KS = T_s + E_ext + E_H + E_xc + E_nn
                    |
                    v
          保存轨道与数值诊断，检查停止条件
                    |
          +---------+---------+
          |                   |
       未满足               已满足
          |                   |
          +--> 下一轮 SCF     v
                         返回 SCF 结果
```

上图的能量是 KS 总能量；固定 H 本征值计算采用第 9 节的独立流程。

SCF 要求轨道产生的密度与该密度生成的有效势相容。当前 `ground_state` 的步骤是：

1. 从当前占据轨道计算电子密度。
2. 使用线性密度混合，或在历史足够时采用 Anderson 混合。
3. 从混合密度构建 Hartree 势、交换关联势及 Kohn–Sham Hamiltonian。
4. 在这个暂时固定的 Hamiltonian 下执行设定次数的 LRRAP 轨道更新。
5. 对保存轨道执行 TT 截断，计算新的输出密度。
6. 用新轨道和新密度计算 Kohn–Sham 总能量。
7. 保存检查点并检查 SCF 停止条件。

线性混合可写为 \(n_{\rm mix}=n_{\rm old}+\alpha(n_{\rm new}-n_{\rm old})\)。当前 Anderson 实现根据密度残差的 Gram 矩阵计算历史权重，加入正则化；系数异常时回退到线性混合。Anderson 混合源自固定点加速方法，参见 [Anderson 的原始论文书目信息](https://projects.csail.mit.edu/jacm/Authors/andersondonaldg.html)。这里的权重限制与回退逻辑是本代码的实现选择。

SCF 总能量按各项直接求和：

\[
E_{\rm KS}=T_s+E_{\rm ext}+E_{\rm H}+E_{\rm xc}+E_{\rm nn}.
\]

存在外场时还计入相应外场势能。Hartree 能量包含 \(1/2\) 因子，以避免电子对相互作用重复计数。

默认 SCF 从第 2 次完整迭代起检查 \(|E_{\rm KS}^{(k)}-E_{\rm KS}^{(k-1)}|<10^{-5}\) Ha。原有 SCF 模块还保留单独的严格配置；本次删除的是固定 Hamiltonian 模块的残差/Gram 收敛判据。

## 8. LRRAP 如何更新占据轨道

```text
当前占据轨道 U、固定的本轮 H、历史搜索方向 P
                    |
                    v
       r_i = H u_i - epsilon_i u_i
                    |
                    v
       z_i = (T + mu I)^(-1) r_i
                    |
                    v
  选取参考 TT -> QR 正交规范化 -> 共同切空间
                    |
                    v
     将 U、Z、P 投影为切空间坐标 D_X、D_Z、D_P
                    |
                    v
     拼接坐标 -> 列归一化 -> SVD -> 保留独立方向
                    |
                    v
     构造投影 Hamiltonian 和重叠矩阵
                    |
                    v
      正交约束下的交替系数优化 (c_i, C_i)
                    |
                    v
       u_i_new = c_i u_i + V C_i
                    |
                    v
          TT 截断、归一化、更新历史方向
                    |
                    v
       必要时执行块 Green 更新与 Ritz 旋转
                    |
                    v
               返回更新后的轨道
```

LRRAP 的文献名称是 **低秩黎曼交替投影**。原方法将多个轨道的修正放在一个共同 TT 切空间中，并交替选择切空间的参考张量，参见 [原论文](https://arxiv.org/abs/1811.11049)。以下步骤对应本代码的 `lrrap_step`，包括额外的 TCI 和块 Green 更新。

**第一步：计算残差及预条件方向。** 对每条轨道形成 \(r_i=Hu_i-\epsilon_i u_i\)，再作用移位动能 Green 算符 \(G_\mu=(T+\mu I)^{-1}\)。代码在当前残差更新阶段采用这一预条件器。

**第二步：选择共同切空间。** 程序轮流以一个占据轨道的 TT 表示为参考张量，构造其切空间。将当前轨道、Green 预条件残差及历史搜索方向投影到这个共同坐标空间。

**第三步：压缩切空间坐标。** 将各坐标列归一化后做 SVD，去除接近线性相关的方向，得到共同切空间基 \(V\)。这里的低秩处理作用于切空间坐标矩阵。

**第四步：优化轨道系数。** 每条新轨道的形式为：

\[
u_i^{\rm new}=c_i u_i+VC_i.
\]

程序在小矩阵中交替优化 \(c_i,C_i\)，目标是降低轨道能量之和，同时满足轨道间的正交约束。原轨道只保留各自的标量系数 \(c_i\)，修正来自共享基 \(V\)。

**第五步：TT 截断与正交化。** 系数确定后，在 QTT 表示中形成新轨道和历史方向，执行 TT 舍入及所需的正交化。这一步相当于将切空间中的候选更新映回受控秩的 TT 表示。

**第六步：可选块 Green 更新。** 达到设定 SCF 间隔，或更新前残差满足相应触发条件时，执行块 Green 更新与 Ritz 旋转，并重置历史搜索方向。该触发条件控制算法步骤，不是本次删除的固定 H 收敛判据。

这条路线仍可用于当前 SCF。固定 Hamiltonian 的默认入口采用下一节的块 Green 迭代，两种入口共享 QTT/TCI 运算，但迭代结构不同。

## 9. 固定 Hamiltonian 的块 Green 迭代

块 Green 迭代在电子结构张量算法中已有明确名称，参见 [Rakhuba 与 Oseledets，§4.1](https://arxiv.org/pdf/1508.07632)。本程序采用周期 FD2 动能和统一正移位 \(\mu\)，属于该思想下的具体离散实现；没有直接照搬文献中的三维连续空间核。

### 9.1 初始试探子空间

本节使用的步骤编号与结果说明中的逐步骤计时对应。`I*` 表示初始化，`K*` 表示一次块 Green 迭代。TCI 构造包含 prrLU；TT 舍入使用 QR/SVD。

```text
初始占据轨道检查点 + 原始固定势 V + 配置
                         |
                         v
 I1  固定势 TCI 与压缩误差诊断；保存诊断数据
                         |
                         v
 I2  构造周期平移 MPO、对角 MPO、Green MPO
                         |
             +-----------+-------------+
             |                         |
         首次计算                    检查点续算
             |                         |
             v                         v
 I3  TCI 包络 + 平面波辅助轨道       载入全部求解轨道
             |                         |
             v                         |
 I4  重叠矩阵 -> 对称正交化 -> TCI 轨道重建
             |                         |
             v                         |
 I5  初始 Rayleigh–Ritz 投影            |
             |                         |
             +------------+------------+
                          |
                          v
 I6  初始残差、重叠矩阵、占据能量诊断；保存检查点
                          |
                          v
        +------ k = 1, 2, ... ------------------------+
        |                                             |
        v                                             |
 K1  对每条轨道：b_i = (mu + epsilon_i - V) u_i        |
        |          点查询 + TCI/prrLU + TT 舍入         |
        v                                             |
 K2  y_i = Green_MPO * b_i                             |
        |          张量缩并 + SVD 截断 + QR/SVD 舍入   |
        v                                             |
 K3  y_i <- y_i / ||y_i||                              |
        |          TT 内积与标量缩放                  |
        v                                             |
 K4  A = Y^H H Y，S = Y^H Y                            |
        |          分块网格求值、差分、矩阵乘法       |
        v                                             |
 K5  A C = S C Lambda                                 |
        |          广义 Hermitian 本征值求解          |
        v                                             |
 K6  U_new = Y C                                      |
        |          线性组合点查询 + TCI/prrLU + 舍入  |
        v                                             |
 K7  残差、Gram 矩阵、E_band = sum_occ f_i epsilon_i   |
        |                                             |
        v                                             |
 K8  保存检查点和日志                                  |
        |                                             |
        v                                             |
     k >= min_iterations 且 |Delta E_band| < 1e-5 Ha ? |
        |                                             |
        +-- 否，且未达 max_iterations ----------------+
        |
        +-- 是 -> converged=true -> 导出 QTT 核心及元数据
        |
        +-- 否，已达 max_iterations -> converged=false
```

上图中 `Y^H` 表示共轭转置。K1、K2、K3 对一条轨道依次完成，再处理下一条轨道；全部轨道完成后进入 K4。每轮有 55 条求解轨道时，K1 调用 TCI 55 次、K6 调用 TCI 55 次，因此每轮共 110 次。K2、K3、默认分块投影 K4、K5 和分块诊断 K7 不调用 TCI。I1 中的势 TCI 用于势压缩诊断和算符初始化；本例 K1、K4、K7 仍读取原始势样本。

载入 50 条占据轨道，再加入 5 条包络平面波作为辅助轨道。辅助轨道的波数、包络周期和振幅均可设置。对整个轨道组正交化，然后执行一次 Rayleigh–Ritz 投影。

“50+5”描述这个算例的子空间维数。函数通过 `states`、`occupied_states` 和 `virtual_wavevectors` 接口支持其他维数；增加辅助轨道不产生额外电子，也不直接给出多电子激发态。

### 9.2 Green 更新

固定 H 本征方程为 \((T+V)u_i=\epsilon_i u_i\)。加入正移位 \(\mu\) 后，可写成固定点形式：

\[
u_i=(T+\mu I)^{-1}\big[(\mu+\epsilon_i-V)u_i\big].
\]

每轮根据上轮轨道和 Rayleigh 商形成：

\[
b_i=(\mu+\epsilon_i-V)u_i,
\qquad \widetilde u_i=G_\mu b_i.
\]

先通过 TCI 构造 \(b_i\)，再用 Green MPO 作用并执行 TT 截断，最后归一化每条 \(\widetilde u_i\)。`green_shift` 就是 \(\mu\)；它改变迭代映射，不改变目标本征方程。

### 9.3 Rayleigh–Ritz 投影

记更新后的轨道组为 \(Y\)。在该子空间中计算：

\[
A=Y^\dagger HY,\qquad S=Y^\dagger Y,
\qquad AC=SC\Lambda.
\]

这是一个广义 Hermitian 本征值问题。本例矩阵大小为 55×55。按 Ritz 值排序后，形成 \(U^{\rm new}=YC\)，并分别用 TCI 重建各轨道。

默认实现按网格块计算 \(A,S\)，不保存完整的 \(N\times55\) 轨道矩阵。动能矩阵用一阶差分内积计算：

\[
A_{ab}^{\rm kin}=\frac{1}{2mh^2}
\sum_j(Y_{j+1,a}-Y_{j,a})^*(Y_{j+1,b}-Y_{j,b}).
\]

它等价于周期 FD2 动能矩阵的投影，并避免在矩阵元中直接相减多个较大的动能项。压缩后的轨道仍会产生数值误差，因此程序继续输出残差和重叠矩阵误差。

### 9.4 占据选择与能量停止条件

将能量最低的 50 条轨道赋予原检查点中的双占据权重，其余 5 条轨道的物理占据数为零。固定 H 下监测的是占据加权本征值和：

\[
E_{\rm band}^{(k)}=\sum_{i=1}^{N_{\rm occ}}f_i\epsilon_i^{(k)}.
\]

当前固定 H 求解器只有以下停止条件：

\[
k\ge k_{\min},\qquad
\left|E_{\rm band}^{(k)}-E_{\rm band}^{(k-1)}\right|<10^{-5}\ \mathrm{Ha}.
\]

能量差还必须是有限、非负数。阈值由 `energy_tolerance` 设置，最少更新数由 `min_iterations` 设置；达到 `max_iterations` 但未满足条件时，`converged=false`。

本征残差和 Gram 矩阵误差不参与固定 H 的停止判定，也不再产生 `residual_converged` 字段。正交化、广义本征值求解和线性独立性检查仍按算法需要执行。

\(E_{\rm band}\) 是占据加权本征值和，SCF 中的 \(E_{\rm KS}\) 是总能量。二者不能混用。相邻能量差也不同于相对参考解的能量误差。

### 9.5 本例的两阶段设置

```text
reproduce 入口：读取两阶段配置
                  |
                  v
启动 Julia 子进程 1：mu = 0.5，至少 3 次更新
                  |
           初始化 I1-I6
                  |
       重复 K1-K8，检查能量变化
                  |
            保存全部轨道
                  |
                  v
启动 Julia 子进程 2：mu = 0.2，至少 6 次更新
                  |
  载入阶段 1 全部轨道，使用同一固定势
                  |
   I1-I2、I6 -> 重复 K1-K8，检查能量变化
                  |
                  v
     导出最终核心、诊断和 bond profile
                  |
                  v
  汇总两个子进程墙钟时间及逐步骤/TCI 记录
```

上图的 bond profile 由保存的核心尺寸在后处理中绘制，绘图不包含在求解子进程计时内。

| 参数 | 第一阶段 | 第二阶段 |
|---|---:|---:|
| 求解子空间维数 | 55 | 55 |
| 占据轨道数 | 50 | 50 |
| Green shift / Ha | 0.5 | 0.2 |
| 最少更新数 | 3 | 6 |
| 能量变化阈值 / Ha | 1e-5 | 1e-5 |
| 轨道／中间张量秩上限 | 48 / 96 | 48 / 96 |
| Green RHS TCI 容差 | 1e-10 | 1e-10 |
| 代数舍入容差 | 1e-12 | 1e-12 |
| Ritz 轨道重建 TCI 容差 | 1e-11 | 1e-11 |

第二阶段从第一阶段保存的全部轨道开始，固定势保持相同。改变 Green shift 后重新记录迭代计数。

原机 dense reference 的第 55/56 态间隙约为 6.14e-3 Ha，第 50/51 态间隙约为 3.05e-6 Ha。扩大求解子空间把部分未占据态纳入联合更新；这可以改变子空间迭代的行为，但不消除占据边界附近的小谱隙。55 不是普适最优维数；其他系统需要按谱结构和实际收敛结果选择辅助轨道数。

## 10. 从 R16 到 R18、R20：周期 Lagrange 网格延拓

```text
粗网格 QTT 轨道（R_coarse）
             |
             v
  校验晶胞、质量、核参数和占据数一致
             |
             v
  周期 Lagrange 插值 + sqrt(h_fine / h_coarse) 缩放
             |
             v
  在细网格索引上调用 TCI -> 细网格 QTT 轨道
             |
             v
         正交化 -> TT 截断
             |
             v
  作为新一轮细网格 SCF 的初始轨道
```

`refine_state` 的实际操作是网格插值延拓。设细粗网格点数之比为 \(q=2^{R_{\rm fine}-R_{\rm coarse}}\)，细网格点写成 \(j=qm+s\)。程序通过周期 Lagrange 插值构造细网格振幅：

\[
u_{\rm fine}(j)=\frac1{\sqrt q}\sum_a\ell_a(s/q)u_{\rm coarse}(m+a).
\]

\(\ell_a\) 是 Lagrange 基函数，粗网格索引按周期边界取模。默认次数为 5，使用 6 个相邻粗网格点。\(1/\sqrt q\) 来自 \(u=\sqrt h\psi\) 的归一化约定。

细网格函数通过 TCI 构造新的 QTT 核心，再正交化和执行 TT 截断，作为细网格 SCF 的初始轨道。R16→R18 和 R18→R20 各有 \(q=4\)。

这里没有使用多个网格上的能量进行 Richardson 外推。将粗网格轨道插值到细网格称为“网格延拓”；只有根据误差阶数组合不同网格结果，才称为“Richardson 外推”。当前实现提供的是前者。

## 11. 实时含时 Kohn–Sham 传播

```text
U(t_n)、固定占据数、外场、Delta t
                  |
                  v
       计算密度 n_1 和中点有效势 V_1
                  |
                  v
    势能半步：exp(-i Delta t V_1 / 2) U   [TCI]
                  |
                  v
    动能整步：Cayley / 复数 Green MPO     [缩并、TT 截断]
                  |
                  v
       重新计算密度 n_2 和中点有效势 V_2
                  |
                  v
    势能半步：exp(-i Delta t V_2 / 2) U   [TCI]
                  |
                  v
      范数与 Gram 漂移检查 -> 观测量 / 检查点
                  |
                  v
     下一时间步，或达到指定步数后返回轨迹
```

时间传播求解：

\[
i\partial_tu_i(t)=H[n(t),t]u_i(t).
\]

占据数保持不变，时间依赖来自轨道、密度反馈以及外加标量势。当前使用绝热局域密度形式的交换关联势。TDKS 传播问题的术语和常见传播子可参见 [Castro、Marques 与 Rubio](https://nano-bio.ehu.eus/files/articles/Castro_JCP_2004_385.pdf)。

设 \(t_{n+1/2}=t_n+\Delta t/2\)，一步执行：

1. 从当前轨道计算密度，在外场时间中点形成有效势 \(V_1\)，执行势能半步 \(u^{(1)}=e^{-i\Delta tV_1/2}u^n\)。
2. 执行 Cayley 动能整步：
   \[
   u^{(2)}=(I+i\Delta tT/2)^{-1}(I-i\Delta tT/2)u^{(1)}.
   \]
3. 从 \(u^{(2)}\) 重新计算密度，以同一个外场时间中点形成 \(V_2\)，执行第二个势能半步。

这种对称安排属于 Strang 算符分裂。其名称来源见 [Strang 的原论文书目](https://klein.mit.edu/~gs/papers/papers.html)。动能步骤用复数移位 Green MPO 实现，等价写为 \(2\mu(T+\mu I)^{-1}u-u\)，其中 \(\mu=2/(i\Delta t)\)。

程序不会在每个时间步自动归一化或正交化，而是检查相对初始轨道的范数和 Gram 矩阵漂移。超出传播误差上限会报错。这是时间传播的数值稳定性控制，与固定 H 的能量停止条件分属不同模块。

用户提供的 [Gross-Pitaevskii-TCI 项目](https://github.com/MarcelNiedermeier/Gross-Pitaevskii-TCI)采用 MPS/TCI 与混合谱二阶 Trotter 时间推进。本项目参考了其计算、保存与后处理的组织方式；本代码动能使用 FD2/Cayley，实现不同于该项目的混合谱传播，也不是时间依赖变分原理（TDVP）算法。

## 12. 保留哪些数值诊断

虽然固定 H 不再用残差和正交性误差决定收敛，程序仍输出：

\[
\eta_i=\frac{\|Hu_i-\epsilon_i u_i\|_2}{\|u_i\|_2},
\qquad \delta_S=\|U^\dagger U-I\|_F.
\]

\(\eta_i\) 是归一化本征残差，单位 Ha；\(\delta_S\) 是重叠矩阵相对单位矩阵的 Frobenius 范数。

日志分别给出全部求解轨道和占据轨道的残差最大值与均方根（RMS）。逐轨道误差、电子数、TT 核心存储量以及键维数也独立记录。它们用于评价结果，不改变当前固定 H 的能量收敛标志。

`audit.py` 可以独立展开保存的 QTT 轨道，并用相同原始势和 FD2 动能重新计算上述量。只有提供同一 Hamiltonian 的参考轨道时，才计算逐轨道本征值误差、占据子空间对齐误差和密度误差。参考解不参与原生 QTT 轨道更新。

占据子空间对齐允许在占据轨道之间做酉旋转，以消除基选择差异。它评价的是占据子空间，并不为近简并本征态指定唯一轨道身份。

## 13. 实测结果与计算复杂度的范围

原机记录中，两阶段分别完成 3 次和 6 次块 Green 更新。两个 Julia 进程的墙钟时间为 269.8116029 s 和 450.6404966 s，合计 **720.4520995 s**。最后占据加权本征值和变化为 **1.5100466655e-6 Ha**；最终最大键维数为 **45**。

该时间包含两个进程的启动、载入、诊断和输出，不含前序 SCF、阶段间空闲、参考解计算、独立核验和绘图。原始测量记录保持不变。本次补充计时后，在同机重新执行相同两阶段配置，子进程墙钟合计 **825.1986364 s**，TCI 总调用 **1103 次**。各步骤的运算、实测时间、逐轮明细和计时范围见 [结果说明](README.md)。

QTT 轨道存储量由核心尺寸决定。但默认固定 H 路线仍保存长度为 N 的势数组，并在整个网格上分块计算投影矩阵与诊断。因此当前端到端实现仍有 O(N) 网格工作量，不能仅凭小键维数声称总时间或峰值内存对网格大小呈对数增长。

## 14. 程序接口与执行顺序

| 源码位置 | 主要函数 | 对应算法 |
|---|---|---|
| `src/numerics.jl` | `cross`, `rounded`, `apply_mpo` | TCI、TT 截断、MPO 作用 |
| 同上 | `lrrap_step`, `coefficients` | 共同切空间更新及约束系数优化 |
| 同上 | `PeriodicRefinement` | 周期 Lagrange 插值 |
| `src/QuanticsKS.jl` | `ground_state` | Kohn–Sham SCF |
| 同上 | `fixed_spectrum`, `simple_green_step`, `spectrum_ritz` | 固定 H 块 Green 迭代及 Rayleigh–Ritz 投影 |
| 同上 | `spectrum_accepted` | 固定 H 能量变化停止条件 |
| 同上 | `refine_state` | 细网格初始轨道构造 |
| 同上 | `propagate` | Strang/Cayley 实时传播 |
| `run.jl` | `main` | 配置、运行、续算、检查点导出 |
| `audit.py` | `verify`, `plot` | 独立数值核验和键维数分布绘图 |

在代码包目录内，可按以下方式使用：

```powershell
# 安装锁定依赖
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# 使用包内输入执行两阶段固定 H 块 Green 迭代
julia --project=. run.jl reproduce Runs/r16

# 从实际参数类型生成完整配置模板
julia --project=. run.jl defaults fixed fixed.toml
julia --project=. run.jl defaults scf scf.toml
julia --project=. run.jl defaults propagation propagation.toml
```

默认固定 H 模板的求解态数为 56，能量停止条件为 1e-5 Ha；本次 R16 两阶段配置在 `r16.toml` 中，求解态数为 55。更换体系时必须同时检查物理模型、占据数、辅助轨道初猜、离散网格以及精度参数。完整可调参数由构造器生成，未知参数会报错。

## 15. 文献来源与当前实现的关系

1. B. N. Khoromskij, *O(d log N)-Quantics Approximation of N-d Tensors in High-Dimensional Numerical Modeling*, Constructive Approximation **34**, 257–280 (2011). [作者机构版本与出版信息](https://www.mis.mpg.de/de/publications/preprint-repository/article/2009/issue-55)。用于 QTT 表示与名称。
2. Y. Núñez Fernández et al., *Learning tensor networks with tensor cross interpolation: New algorithms and libraries*, SciPost Physics **18**, 104 (2025). [论文](https://arxiv.org/abs/2407.02454)。用于 TCI、prrLU 和相关软件术语。
3. M. Rakhuba, A. Novikov, I. Oseledets, *Low-rank Riemannian eigensolver for high-dimensional Hamiltonians*, Journal of Computational Physics **396**, 718–737 (2019). [论文](https://arxiv.org/abs/1811.11049)。用于 LRRAP、共同切空间与受约束系数优化的来源说明。
4. M. Rakhuba, I. Oseledets, *Grid-based electronic structure calculations: The tensor decomposition approach*, Journal of Computational Physics **312**, 19–30 (2016). [论文](https://arxiv.org/abs/1508.07632)。用于 Block Green iteration 的专业名称与电子结构背景；其三维 Tucker 实现不等同于本项目一维 QTT 实现。
5. D. G. Anderson, *Iterative Procedures for Nonlinear Integral Equations*, Journal of the ACM **12**, 547–560 (1965). [原期刊书目](https://projects.csail.mit.edu/jacm/Authors/andersondonaldg.html)。用于 Anderson 固定点加速的来源。
6. G. Strang, *On the Construction and Comparison of Difference Schemes*, SIAM Journal on Numerical Analysis **5**, 506–517 (1968). [作者书目](https://klein.mit.edu/~gs/papers/papers.html)。用于 Strang 算符分裂的名称与来源。
7. A. Castro, M. A. L. Marques, A. Rubio, *Propagators for the time-dependent Kohn–Sham equations*, Journal of Chemical Physics **121**, 3425–3433 (2004). [作者机构全文](https://nano-bio.ehu.eus/files/articles/Castro_JCP_2004_385.pdf)。用于实时 TDKS 和传播子术语。
8. M. Niedermeier et al., *Solving the Gross-Pitaevskii equation on multiple different scales using the quantics tensor train representation*, Physical Review Research **8**, 023006 (2026). [论文及出版信息](https://arxiv.org/abs/2507.04262)；[作者代码](https://github.com/MarcelNiedermeier/Gross-Pitaevskii-TCI)。用于用户指定的软件组织参考，未据此宣称本代码具有该论文的性能。
