# 百度路网转化（Baidu-to-SUMO）技术设计文档

> **文档地位**：本文档（`docs/DESIGN.md`）为本仓库**唯一权威技术设计文档（Single Source of Truth）**。项目规范与红线见根目录 [AGENTS.md](../AGENTS.md)。任何架构演进必须原地更新本文档，严禁另建平行设计文档。

---

## 1. 设计目标与产品边界

### 1.1 核心目标

以 `/Volumes/USB DISK/爬取百度路网/output/广州市/` 中的广州市百度车道级矢量数据为输入，通过**纯机器学习 / 深度学习 / 强化学习（ML / DL / RL）**管线，在**训练全程零人工标注**条件下，自动重建可直接用于 SUMO 仿真且在百度原始底图上精准叠加可视化的高精度路网（`*.net.xml` 及对应标准化有向车道图产物）。

### 1.2 范围与输出边界

| 维度 | Phase 1 明确定义 | 未来扩展（Phase 2 / 3）预留 |
| --- | --- | --- |
| **空间视野（ROI）** | **支持任意尺寸与形状的框选范围**（面向未来人工框选任意路口/局部区域，不写死固定视野）；单窗原生支持至 `204.8 m × 204.8 m`（净 ROI `172.8 m`），超大选区于 Phase 1 即支持重叠分块与端口拼接；单平面交叉口优先验证 | 任意大范围走廊与片区（复用同一固定米制分辨率与 `roi_entry/exit_port` 端口匹配拼接框架） |
| **保留对象** | 具有明确车道标线、导向箭头、渠化/分隔特征的规范机动车道（含公交/可变/潮汐车道属性）、交叉口内部连接与人行横道 | 立体交叉口多层道路（通过节点/控制点层序 `z` 扩展与输入显示层序通道扩展） |
| **摒弃对象** | 停车场内部通道、停车位区、小区/园区内部路、非机动车道、港湾式公交站及无标线细小支路（**完全由学习模型识别剔除**，零硬编码面积/长度/距离阈值） | 保持一致 |
| **输出实体** | **标准化有向车道图 $G = (V, L, K_{\text{conn}}, C, E_{\text{adj}}, E_{\text{from}}, E_{\text{to}}, E_{\text{succ}}, E_{\text{cross}}, E_{\text{cross\_node}})$**：显式节点与路口轮廓 `V`、等宽有向车道段 `L`（含权限与变道标志）、成对内部连接轨迹 `K_conn`、人行横道 `C` 与完整拓扑/归属矩阵 | 节点与控制点扩展层序/三维坐标 `z` 及跨块端口连接边 |
| **不建模对象** | **模型不预测信号灯与信号配时**（交付产物 `site.net.xml` 为无信号优先权几何+拓扑基础路网，不含模型预测的 `.tll.xml` / 相位方案；面向高饱和多岔仿真时需由下游交通控制/配时系统补充信号）。*注：评测侧既在交付版 `site.net.xml` 上执行无冲突稀疏单 OD 连通性核验，又在 RL 奖励与饱和冲突流评测的**临时仿真测试台副本 `eval_sim.net.xml`** 中仅对模型输出的 `junction_core` 节点启用 `netconvert --tls.set=<junction_core_ids>` 自动生成防死锁测试台配时，绝不替代交付网且绝不误给路段 `midblock_node` 建灯。* | 按需作为独立下游模块接入 |
| **标注与评测策略** | **训练与验证全程 100% 零人工标注**（参数化合成预训练 + 真实广州物理自监督 + SUMO-in-the-Loop RL）；**评测侧**执行「客观指标 + 强制底图叠加视觉终审」，并预留独立只读的 **Eval-Only 人工核验审计集（20–40 口）** 接口，用于检测纯自参考指标无法暴露的车道数/转向系统性偏差（绝不参与训练与调参） | 保持训练与评测严格隔离 |

---

## 2. 总体技术架构总览

系统由**双编码器多模态感知主干**、**Dense-to-Sparse 节点/车道/连接分层解码器**、**约束图解码与确定性 SUMO 序列化闭环**以及**三层无人工标注学习飞轮**四大部分组成：

```text
[任意框选 ROI 的百度 GPKG 原始矢量（统一投影至 EPSG:32649 UTM 49N）]
       │
       ├──► 固定分辨率 (0.20 m/px) 抗锯齿覆盖率 + 截断 UDF 多通道 BEV (36ch) ──► 高保真轻 Stem + 主干 + 4 层 FPN {F_2, F_4, F_8, F_16}
       │                                                                                       │
       └──► 箭头/标线/斑马线/岛屿等 稀疏矢量 Token 化 ────────────────────► 矢量编码器 (Polyline + RPE Set Transformer)
                                                                                               │
                                                                       [双向 Deformable Cross-Attention 融合（MPS 原生双线性采样）]
                                                                                               │
                   ┌───────────────────────────────────────────────────────────────────────────┴───────────────────────────────┐
                   ▼                                                                                                           ▼
    [稠密辅助感知头（挂接 F_2 / F_4）]                                                                             [Dense-to-Sparse 分类别 Query 提议]
    - 有效有标线道路掩码 M_valid                                                                                    - 节点 Query (P_q=64)
    - 中心线热图 / SDF & 方向场 (cosθ, sinθ)                                                                       - 路段车道层级 Query (N_q=128)
    - 局部辅助宽度场 W_map（仅作训练辅助监督）                                                                        - 人行横道 Query (M_q=24)
    - 停止线 / 分合流 / 路口中心 / 斑马线锚点热图                                                                              │
                                                                                                                               ▼
                                                                                                           [6 层解耦迭代矢量解码器（Intra/Inter-Instance）+ GNN 拓扑与成对连接头]
                                                                                                           - 节点 V（坐标、类型、路口轮廓、port_embed）
                                                                                                           - 路段车道 L（32点、标量宽、角色、权限、禁变道标志）
                                                                                                           - 人行横道 C & 关系 E_adj / E_from / E_to / E_succ / E_cross
                                                                                                           - 成对连接轨迹解码头 → 内部连接 K_conn（8点，动态≤256）
                                                                                                                               │
                                                                                                                               ▼
                                                                                                           [结构化约束图解码（含非零度与保序无交叉）+ Layer 3 RL 离散拓扑精修]
                                                                                                           （多对偏好迭代 DPO 主力 + 局部优势结构化 GRPO 可选）
                                                                                                                               │
                                                                                                                               ▼
                                                                                                           [Canonical Directed Lane Graph G]
                                                                                                            ──► 零启发式 1:1 转写 SUMO plain XML（含显式边/节点形与禁自动补连）
                                                                                                            ──► netconvert（保真参数 + 往返拓扑/几何一致性核验）+ sumo 仿真 + 底图叠图
```

---

## 3. 输入表征与任意视野（Arbitrary ROI）适配设计

### 3.1 统一米制投影与任意选区处理

为同时满足**任意框选视野（不扭曲车道物理比例）**与 **Phase 3 跨块无缝拼接及 GIS 精准叠图**，制定如下坐标与视野协议：

1. **全市统一米制投影（EPSG:32649 / UTM 49N）**：
   - 经实测广州全库覆盖范围为 `112.953°E–114.053°E, 22.546°N–23.936°N`，与原数据工程自带投影基准一致，全系统统一采用 **UTM Zone 49N（`EPSG:32649`，中央经线 `111°E`）** 作为全局米制平面坐标。按 WGS84 椭球实算，全广州市尺度因子在 `1.0001–1.0008` 之间（市中心约 `+0.03%`，最东端 `~114.05°E` 约 `+0.08%`，折合 `3.5 m` 车道宽差异仅 `~1–3 mm`）；由于百度底图与输出路网处于同一投影系，叠图无投影偏差，且彻底消除各 ROI 独立 ENU 质心导致的子午线收敛角与原点不统一问题。
   - 单窗推理与 SUMO 序列化时，减去窗口左下角取整后的全局偏移原点 `(X_0, Y_0)` 以保证 `float32` 数值精度，并在输出的 `.nod.xml` 与 `site.net.xml` 的 `<location netOffset="-X_0,-Y_0" projParameter="+proj=utm +zone=49 +datum=WGS84 +units=m +no_defs" .../>` 中完整落盘，确保与原始 WGS84 百度底图及相邻分块 100% 精确对齐。
2. **固定米制分辨率**：BEV 栅格始终采用固定网格步长 **`0.20 m/px`**（1 米 = 5 像素，单条 `3.2 m` 车道宽 16 像素），严禁将任意选区缩放拉伸到固定像素尺寸。
3. **显式选区掩码与边缘缓冲**：
   - 读取空间要素时在用户选区外自动扩展 **`16 m`** 上下文缓冲，避免选区边缘标线与导向箭头被硬截断；
   - 输入中包含 `roi_mask` 通道（选区内 `1`，缓冲上下文区 `0`），模型仅在选区内生成路网，并在选区边界自动产出 `roi_entry_port` / `roi_exit_port` 节点。
4. **任意尺寸推理与 Phase 1 即备的分块拼接**：
   - 经 `开创大道_WGS84.gpkg` 实测，广州主干交叉口核心斑马线外包跨度可达 `60–100 m`，含上下游导向箭头与展宽段的跨度常达 `130–200 m`。仅靠 `128 m` 净 ROI 易截断上游展宽段，因此扩大单窗容量并将分块拼接提前至 Phase 1 必备能力：
   - **训练期**：采用多尺寸随机窗口（`512×512` 至 `896×896`，对应 `102.4 m` 至 `179.2 m`）叠加随机形状/长宽比 `roi_mask` 增强；
   - **推理期**：含两侧 `16 m` 缓冲总跨度 `≤ 204.8 m × 204.8 m`（`1024×1024`，对应净 ROI 上限 `172.8 m × 172.8 m`）时整窗单次前向（padding 至 32 倍数）；超过单窗上限的超大路口或任意大选区，按固定重叠步长分块推理，利用 `V` 中 `roi_entry_port` / `roi_exit_port` 节点的统一全局 UTM 坐标与学习得到的 `port_embed ∈ R^D` 通过学习型端口匹配头跨块拼接（M0 完成确定性分块图拼接单测，M1/M2 在超大选区上联调验收，Phase 3 直接平滑复用）。

### 3.2 双路输入表征与双编码器融合

#### 分支 A：抗锯齿多通道米制 BEV 栅格（36 通道）

经对广州全市 4,153,856 条 `南沙道路要素`、`其他原生区域面`、`公交站点位` 及 `开创大道_WGS84.gpkg` 的实测核查：
- 百度标线漆宽实测为 `0.116 m – 0.239 m`（白实线中位宽 `0.116–0.192 m`，白虚线 `0.180 m`，黄实线 `0.205–0.239 m`），在 `0.20 m/px` 下仅占 `0.58–1.20` 像素。**硬二值光栅化必然产生丢线与严重锯齿**；
- 因此采用**选择性抗锯齿光栅化策略（兼顾亚像素保真与 CPU 吞吐）**：仅对窄线/窄漆面通道（白/黄标线、`∂road_surface`、`separation_structure_path`、`greenway_boundary`、`斑马线白色条纹`）执行 `4×4` 子像素超采样面积覆盖光栅化（或若 M0 基准测试接近 CPU 吞吐预算，切换至零超采样的**解析亚像素带宽线段覆盖率光栅化**作为高性能等价实现），其余宽面要素（`roi_mask`、`road_surface`、`channelization_area`、`traffic_island`、`crosswalk_area`、`parking_*`、`arrow_*`、`z_order_*`）直接采用快速 `1×` 精确光栅化；并附加全标线截断无符号距离场（Truncated UDF）与标线无向主轴场 `(cos 2φ, sin 2φ)`，使窄标线在 `0.20 m/px` 下保留完整的亚像素质心与朝向信息；
- **源数据分类与非机动编码事实**：
  1. 广州全库 45 个 `分类代码` 中**无独立 `stop_line`（停止线）、`left_turn_waiting_area`（左转待转区）、`curb`（路缘）或 `yellow_double_solid` 类**——停止线与待转区以几何形态存在于 `white_solid_stroke` / `white_dashed_stroke` 中，双黄线为平行 `yellow_solid_stroke`，导流斜纹已完整记录于 `channelization_area`（中位宽 `0.46 m`），路缘可由合并后的 `road_surface` 外轮廓 `∂road_surface` 确定性提取；
  2. **非机动车道与公交港湾实测事实**：`其他原生区域面` 仅含 `render_reflection_surface` 与 `parking_background_surface`，**全库无独立的非机动车道面要素**。在 `开创大道_WGS84.gpkg` 实测中，全部 `17/17` 个 `bicycle_lane_symbol` 均与 `road_surface` 相交（与机动车道箭头、白虚线共享同一 `road_surface` 多边形），距 `separation_structure_path` 中位仅 `0.23 m`、距实线标线中位仅 `0.77 m`；`公交站点位` 亦直接落于 `road_surface` 边界内（中位距离 `0.00 m`）。因此非机动车道与港湾式公交站的识别与剔除必须纳入 Layer 1 合成负样本与 Layer 2 防退化覆盖设计（见第 5 节）。

完整 36 通道定义如下（每一通道均严格映射至实测存在的 GPKG 图层与 `分类代码`）：

| 通道组 | 通道数 | 通道内容与编码方式 | 对应真实 GPKG 图层 / `分类代码` |
| --- | :---: | --- | --- |
| **选区与基础面** | 5 | `roi_mask`、道路面 `road_surface`、**路缘/道路外边界 `∂road_surface`**（确定性轮廓线）、导流区/斜纹 `channelization_area`、安全岛 `traffic_island` | `南沙道路要素`（`road_surface`、`channelization_area`、`traffic_island`） |
| **白色标线** | 5 | 白虚线、白实线（含纵向实线与横向停止线）、白双实线、白左虚右实、白左实右虚（抗锯齿覆盖 `[0,1]`） | `white_dashed_stroke` + `white_dashed`、`white_solid_stroke`、`white_double_solid`、`white_left_dashed_right_solid`、`white_left_solid_right_dashed` |
| **黄色与专用标线** | 6 | 黄实线（含双黄线）、黄左实右虚、黄双虚线、公交专用车道、可变车道、潮汐车道（抗锯齿覆盖 `[0,1]`） | `yellow_solid_stroke`、`yellow_left_solid_right_dashed`、`yellow_double_dashed`、`bus_lane_marking`、`variable_lane_marking`、`reversible_lane_marking` |
| **标线亚像素几何增强** | 3 | 全标线截断距离场 `UDF_marking`（截断半径 `2.0 m` 归一化至 `[0,1]`）、标线无向主轴场 `(cos 2φ, sin 2φ)`（提供标线全局无向朝向先验，供网络结合车道行驶方向 $\theta$ 联合判别横向停止线与纵向分道线） | 上述所有白/黄标线联合计算 |
| **人行横道与隔离** | 3 | 斑马线范围 `crosswalk_area`、斑马线白条纹、隔离设施路径 | `南沙道路要素`、`斑马线白色条纹`、`隔离设施路径`（`separation_structure_path`） |
| **显式负样本与非机动** | 3 | 各类停车位面与停车背景、停车场内部通道符号、非机动符号与慢行边界 | `parking_space_surface` / `onstreet_parking_space` / `offstreet_parking_space` / `service_area_parking_marking` / `其他原生区域面.parking_background_surface`；`parking_bidirectional_symbol` / `parking_oneway_symbol`；`bicycle_lane_symbol` / `greenway_boundary` |
| **导向箭头密集先验** | 9 | 箭头存在掩码、箭头精确朝向 `(cos θ, sin θ)`、**6 个独立二值转向通道 `[L, S, R, U, FL, FR]`**（消除单通道位掩码的有序偏差） | 全部 17 类 `arrow_*` + `uturn_allowed_symbol` |
| **Phase 2 预留（显示层序）** | 2 | **相对显示层序均值 `z_order_mean`、相对显示层序跨度 `z_order_span`**（App 原始相对显示层序，**非实测海拔高程**，与 GPKG `Z语义` 严格一致；Phase 1 平面场景置 0） | `道路面_三维` / `源显示Z代表` |

#### 分支 B：关键实体稀疏矢量 Token 集合

为弥补栅格在长距离连贯拓扑与精确组合箭头语义上的不足，将 ROI 内关键实体编码为稀疏矢量 Token（单窗上限 `M_vec ≤ 512`）：
- **导向箭头 Token**：质心 `(x, y)`、精确朝向 `(cos θ, sin θ)`、尺寸 `(w, l)`、6 维多标签向量 `[L, S, R, U, FL, FR]`（无损保留所有组合转向语义）；
- **核心标线与路缘折线 Token**：对白实/白虚/黄线/专用标线/隔离护栏/路缘轮廓提取等距采样中心折线（每条 `P=8` 点，含 `(x, y)`、切向 `(dx, dy)`、类别嵌入与实测局部漆宽）；
- **斑马线、岛屿与非机动/公交站上下文 Token**：斑马线范围/白条主轴、安全岛/导流区外轮廓采样点，以及 `bicycle_lane_symbol`、`greenway_boundary`、`公交站点位` 点/轮廓上下文 Token（含类别与尺寸嵌入，显式辅助学习剔除非机动车道与港湾式公交站）。

#### 双编码器融合（Dual-Encoder Fusion）

1. **高保真 BEV 编码器（尺度严格一致、适配大窗显存）**：
   - 放弃 `ConvNeXt` 默认的激进 `4×4 stride=4` patchify stem（会一步下采样至 `0.80 m/px` 抹平 `0.15 m` 标线细节），改用 **轻卷积高保真 Stem**（`3×3 stride=1` + `3×3 stride=2` 残差块，输出 `1/2` 尺度高分辨率特征 `C_2`，`0.40 m/px`），后接轻量主干阶段（`ConvNeXt-Tiny` 后 3 阶段或 `ResNet34` 对应阶段，输出 `{C_4, C_8, C_16}`）；
   - 经 **4 层 FPN** 自顶向下融合，输出严格对齐的多尺度特征 `{F_2, F_4, F_8, F_16}`（对应 `0.40 m / 0.80 m / 1.60 m / 3.20 m` 步长），其中 `{F_4, F_8, F_16}` 统一投影至 `D = 256` 维，高分辨率 **`F_2` 采用轻量投影 `D_2 = 128` 维**（将 `1024×1024` 单窗推理下的 `F_2` 激活显存减半，适配 M4 Pro 24 GB 统一内存）。
2. **矢量编码器**：轻量 PointNet/Polyline-MLP 聚合点序列为实体 Token（`D=256`），经 2 层带相对位置偏置（RPE）的 Self-Attention 建模长距离矢量上下文。
3. **双向跨模态融合（2 层，MPS 原生算子）**：
   - 对 `{F_4, F_8, F_16}` 与矢量 Token 执行双向 **Deformable Cross-Attention**（`Vector-to-BEV` + `BEV-to-Vector`），输出增强后的多尺度多模态特征 `F_fused = {F_2, F_4', F_8', F_16'}`（其中 `F_2` 融合 `F_4'` 上采样语义，专供稠密细标线感知头）与矢量记忆 `V_fused`。
   - **Apple MPS 算子适配**：经实测本机 PyTorch 2.8.0 MPS 无原生 CUDA `MSDeformAttn` 且 `aten::grid_sampler_2d_backward` 尚未支持 MPS（直接调用会报错或回退 CPU）。因此本项目的 Deformable Cross-Attention 统一采用 **纯 MPS 原生张量算子（基于 `floor` + `gather` + 显式双线性插值权重）** 实现可微多尺度采样，保证前向与反向 100% 驻留 MPS 零 CPU 回退。

---

## 4. 统一输出表征、约束解码与确定性 SUMO 序列化

### 4.1 标准化输出表征（Canonical Directed Lane Graph）

为实现与 SUMO `plain XML`（`.nod.xml` / `.edg.xml` / `.con.xml`）严格 **1:1 确定性无启发式映射**，输出表征显式建模节点、等宽车道段、成对连接轨迹、人行横道及完整归属与拓扑矩阵：

$$G = (V, L, K_{\text{conn}}, C, E_{\text{adj}}, E_{\text{from}}, E_{\text{to}}, E_{\text{succ}}, E_{\text{cross}}, E_{\text{cross\_node}})$$

1. **显式节点实体集合 `V = {v_1, ..., v_P}`**（单窗上限 **`P_max = 64`**；全部节点均严格对应底图上视觉可观测的拓扑或属性事件，取消一切依赖裁切框起点的固定弧长伪切分节点）：
   - `pos ∈ R^2`（Phase 2 扩展为 `R^3`）：节点米制坐标 `(x, y)`；
   - `node_type ∈ {junction_core, split_merge, midblock_node, roi_entry_port, roi_exit_port, road_continuation}`：节点语义类型（对应 SUMO `<node>` 及 Phase 1/3 边界端口；其中 `midblock_node` 专用于路段中间掉头开口及非路口核心区人行横道，`road_continuation` 仅用于底图可见的专用道起止或实虚禁变道属性切换断面）；
   - `polygon ∈ R^{16×2}`：显式路口闭合轮廓多边形（对 `junction_core` 激活并施加轮廓-接入车道端点一致性监督，直接映射 `<node shape="..."/>`；`split_merge` / `midblock_node` / `road_continuation` 在序列化阶段由其接入车道的精确外轮廓端点确定性构造闭合 `<node shape="..."/>`，防止 `netconvert` 裁切或拉伸车道端点）；
   - `port_embed ∈ R^D`：端口匹配特征嵌入（仅对 `roi_entry_port` / `roi_exit_port` 在超大选区跨块拼接时使用，与 3.1 节完全对齐）；
   - `score ∈ [0, 1]`：存在置信度。
2. **有向机动车道段实体集合 `L = {l_1, ..., l_N}`**（**仅含非路口内部、非渐变过渡区的恒宽路段车道段**，单窗上限 **`N_max = 128`**）：
   - **`points ∈ R^{32×2}`（`K_lane = 32`，Phase 2 扩展为 `R^{32×3}`）**：按行驶方向严格有序的 `32` 个等弧长米制中心线控制点（共 31 个间隔）。将“控制点几何密度”与“图节点切分”彻底解耦——取消 `37.5 m` 固定弧长人为切段后，整条 `60–90 m` 连续引道无需插入任何不可观测伪节点即可保持 `1.9–2.9 m` 高精度点距，既消除 Hungarian 匹配目标的裁切框随机噪声与 SUMO 伪路口禁变道伪影，又将大路口所需车道段数量降低约 `40%–50%`，使 `N_max = 128` 充裕覆盖 6 岔大展宽路口；
   - **`width ∈ R^+`（标量宽度）**：对应 SUMO `<lane width="..."/>` 严格 1:1 映射。当路段发生展宽/缩减渐变段 `[s_start, s_end]` 时，上游恒宽车道段统一在渐变起点 `s_start` 终止，下游恒宽车道段统一在渐变终点 `s_end` 起始（展宽新增车道严格从达到完整标称宽度的 `s_end` 断面起存在，缩减消失车道严格在开始收窄前的 `s_start` 断面终止）；渐变段内部不放置变宽 `L` 车道段，而是由单个 `split_merge` 节点及其成对过渡轨迹 `K_conn` 平滑桥接，彻底消除标量宽度自相矛盾与序列化启发式；
   - `role ∈ {approach, exit, mid_segment, slip_turn, uturn_bay}`：车道段语义角色；
   - **`permission ∈ {general, bus_only, variable, reversible}`**：车道通行权限类别（直接支持广州公交专用道 `allow="bus"` 与可变/潮汐车道属性落盘）；
   - `lane_change_flags = (allow_change_left, allow_change_right) ∈ {0, 1}^2`：左右变道许可标志（学习识别白实线/双实线禁变道，按 SUMO 1.28.0 实测规范映射为 `changeLeft="authority"` / `changeRight="authority"`）；
   - `is_stop_line_end ∈ {0, 1}`：终点是否为进口停止线断面；
   - `score ∈ [0, 1]`：存在置信度。
3. **成对连接轨迹实体集合 `K_conn`**（与路段车道解耦，动态支持单窗最多 `256` 条，彻底解决大路口容量不足）：
   - 针对 `E_succ` 激活的合法有向车道段对 `(l_i, l_j)`（统一覆盖跨 `junction_core` 的路口内左/直/右/掉头转向连接、跨 `split_merge` 渐变段 `[s_start, s_end]` 的平滑分合流过渡轨迹，以及跨 `midblock_node` 的路段提前掉头轨迹），直接预测 **`points ∈ R^{8×2}`（Phase 2 为 `R^{8×3}`）**：首尾两点精确锁定于 `l_i` 终点与 `l_j` 起点，中间 6 个自由控制点描绘无压岛平滑过渡轨迹，1:1 映射 SUMO `<connection shape="..."/>`；
   - **左转待转区表征约定**：因 Phase 1 不耦合两阶段信号灯相位，广州高频的路口内**左转待转区不单独建为 `L` 车道段或伪节点**；进口左转车道 `L` 终止于上游主停止线断面，待转区走廊几何自然吸收并由对应左转 `K_conn` 轨迹贴合引导线平滑穿过。
4. **人行横道实体集合 `C = {c_1, ..., c_M}`**（单窗上限 `M_max = 24`）：
   - `endpoints ∈ R^{2×2}`：斑马线中心轴起止端点 `(p_start, p_end)`；
   - `width ∈ R^+`：斑马线标量宽度；
   - `score ∈ [0, 1]`：存在置信度。
5. **学习得到的拓扑与归属关系矩阵**：
   - **横向相邻矩阵 `E_adj ∈ [0, 1]^{N×N}`**：`E_adj[i, j] = 1` 表示车道段 `j` 紧邻在同向车道段 `i` 的**左侧**（遵循 SUMO 从最右侧 `lane_0` 向最左侧编号约定），且属于同一 `<edge>` 断面；
   - **车道端点→节点归属矩阵 `E_from, E_to ∈ [0, 1]^{N×P}`**：分别指示每条车道段 `l_i` 的起点归属节点 `from(l_i) ∈ V` 与终点归属节点 `to(l_i) ∈ V`；
   - **纵向连接矩阵 `E_succ ∈ [0, 1]^{N×N}`**：`E_succ[i, j] = 1` 表示车辆可从 `l_i` 终点合法驶入 `l_j` 起点（统一表达展宽分流、缩减合流、渠化右转、路段提前掉头与路口内部转向）；
   - **人行横道归属与跨接矩阵 `E_cross_node ∈ [0, 1]^{M×P}`、`E_cross ∈ [0, 1]^{M×N}`**：分别指定人行横道 `c_m` 归属的合法节点（允许绑定至 `junction_core` 或 `midblock_node`，统一支持路口核心区斑马线、渠化右转道斑马线与路段中段人行横道）及跨越的机动车道段，直接映射 SUMO `<crossing node="..." edges="..."/>`。

### 4.2 Dense-to-Sparse 分层解码与约束图解码器

1. **阶段 A — 稠密辅助感知头（挂接在 `F_2` 与 `F_4'`，与 3.2 节严格对齐）**：
   - 有效有标线机动车道掩码 `M_valid`（二分类分割：规范机动车道 = 1，无标线支路/停车场/内部路/非机动车道/港湾式公交站/背景 = 0）；
   - 车道中心线热图 `H_center` 与有符号距离场 `SDF_center`；
   - 稠密行驶方向场 `(cos θ, sin θ)` 与局部逐点辅助宽场 `W_map`（仅用于训练期像素级辅助监督与提议特征，不作为 SUMO 输出契约）；
   - 节点与关键锚点热图：路口核心中心、分合流渐变段中心、路段功能节点（掉头开口 / 非核心区斑马线中心）、ROI 边界端口、停止线锚点、斑马线中心锚点。
2. **阶段 B — Dense-to-Sparse 分类别 Query 提议**：
   - 从稠密感知特征与 `V_fused` 高响应位置，通过可学习 Top-K 评分器分别初始化三类异构 Query：节点 Query（`P_q = 64`）、路段车道层级 Query（`N_q = 128`，按 MapTRv2 展开为 `instance_q + point_q` 共 **`32`** 个点 Query）与斑马线 Query（`M_q = 24`）。
3. **阶段 C — 6 层解耦联合迭代矢量解码器（MapTRv2 Decoupled Hierarchical Self-Attention）**：
   - 为避免全点级联合自注意力（`128 × 32 = 4096` 点，`O(4096²)`）在 MPS 上触发显存与带宽瓶颈，每层采用**解耦层级自注意力**：
     1. **实例内点级自注意力（Intra-Instance Self-Attention）**：每条路段车道仅在自身 `32` 个控制点间交互（`128 × 32²`），高效建模单条车道的平滑曲率与等弧长几何约束；
     2. **跨实体实例级自注意力（Inter-Instance Self-Attention）**：对车道点特征做池化聚合为实体 Token 后，与节点、斑马线全部 `64 + 128 + 24 = 216` 个实体 Token 执行全局自注意力（`216²`），计算与显存较全点联合注意力降低 **`> 20×`**；
     3. 随后依次执行：`对 {F_4', F_8', F_16'} 的多尺度 MPS 原生 Deformable Cross-Attention` → `对 V_fused 的稀疏矢量 Cross-Attention` → `FFN` → 迭代更新节点坐标/轮廓、车道 32 点偏移 `Δ(x, y)`、标量宽度 `Δw`、属性分类与斑马线几何。
   - 训练期分别对 `V`、`L`、`C` 执行独立 **Hungarian 集合匹配**（有向折线固定行驶方向，消除方向模糊）；对 `junction_core` 节点同步施加**路口轮廓-接入车道端点一致性损失**（要求 `polygon` 边界贴合其接入车道的起止横断面角点）。
4. **阶段 D — GNN 拓扑关系头 + 成对连接轨迹解码头（TopoNet 风格）**：
   - 将最终节点特征 `H_V`、车道特征 `H_L`（拼接起终点坐标/切向/标量宽）与斑马线特征 `H_C` 送入 2 层异构图注意力层，由双线性/MLP 边头并行输出 `E_adj`、`E_from`、`E_to`、`E_succ`、`E_cross`、`E_cross_node` 的概率矩阵；
   - **成对连接轨迹解码头（Pairwise Connection Trajectory Decoder）**：对训练期真值正对 / 推理期 `E_succ` 激活的合法有向对 `(l_i, l_j)`，拼接 `(h_i, h_j, end_i, tangent_end_i, start_j, tangent_start_j)`，在起终点连线初始插值上执行 2 层多尺度 Deformable Cross-Attention，回归中间 6 个内部控制点偏移，输出 `K_conn` 的 8 点精确内部轨迹。
5. **阶段 E — 结构化约束图解码器（Constrained Maximum-Weight Graph Decoder）**：
   神经网络输出的概率矩阵直接阈值化可能产生分叉、成环、节点不一致或局部死胡同，因此推理末端通过**确定性图论组合优化**（纯数学约束优化，属于 `AGENTS.md` 3.3 节允许的图论约束解码，零几何启发式阈值）将概率矩阵解码为 100% 拓扑合法的离散有向车道图：
   - **`E_adj` 编组解码**：对置信度通过筛选的有效车道段 `L_act`，以对数似然比 $\log \frac{E_{\text{adj}}[i,j]}{1 - E_{\text{adj}}[i,j]} - \tau_{\text{adj}}$ 为边权，求解**无环最大权有向路径覆盖**（约束每条车道出度 `≤ 1`、入度 `≤ 1`、无环，等价于带破环约束的最大权二分匹配 / 小型 ILP，`N ≤ 128` 毫秒级精确求解）。每条极大有向链确定性编组为一个 SUMO `<edge>`，链序即为从最右侧向最左侧严格递增的机动车道顺序；
   - **联合 `(from, to)` 节点归属解码**：对每条编组出的 `<edge>` 集合 $S_e \subseteq L_{\text{act}}$，最大化组内联合节点对数似然 $\arg\max_{u \neq v} \sum_{i \in S_e} (\log E_{\text{from}}[i, u] + \log E_{\text{to}}[i, v])$，严格保证同一 `<edge>` 下所有并行车道共享唯一的 `(from_node, to_node)`；
   - **`E_succ` 连通与保序合法化解码**：以 $\log \frac{E_{\text{succ}}[i,j]}{1 - E_{\text{succ}}[i,j]} - \tau_{\text{succ}}$ 为边权，在满足以下三项纯组合约束的可行集上求解最大权连接子图（单节点变量极少，动态规划 / 小型 ILP 毫秒级精确求解）：
     1. **节点衔接约束**：仅允许 `to(l_i) == from(l_j)` 的候选对；
     2. **内部节点非零度连通约束**：对终点为非 `roi_exit_port` 内部节点且该节点存在有效出向候选车道的进口车道段强制出度 `≥ 1`；对起点为非 `roi_entry_port` 内部节点且该节点存在有效入向候选车道的出口车道段强制入度 `≥ 1`；
     3. **同 `(edge_in, edge_out)` 单调保序无交叉约束**：同一进口 `<edge>` 驶入同一出口 `<edge>` 的连接对严格满足单调不交叉（$i_1 < i_2 \implies j_1 \le j_2$）。
     *防遮蔽纪律*：评测阶段**强制双轨记录“约束解码前原始阈值输出”与“约束解码后输出”两套拓扑指标**，严防组合优化器掩盖神经网络自身预测错误；
   - **`E_cross` 与 `E_cross_node` 合法化解码**：每条有效斑马线 `c_m` 在 `node_type ∈ {junction_core, midblock_node}` 的有效节点集合上取 $\arg\max_p E_{\text{cross\_node}}[m, p]$ 绑定至所属节点，并在与该节点关联的 `<edge>` 集合上解码 `E_cross` 生成目标 `edges` 列表。

### 4.3 确定性 SUMO 序列化器与 `netconvert` 编译契约（经 SUMO 1.28.0 逐项实测核验）

序列化模块 `src/sumo_io/` 仅执行 `G → SUMO plain XML` 的纯语法转写与编译后往返核验，严禁任何几何补猜或拓扑修补；所有默认属性与枚举映射统一收口于 `configs/sumo_export.yaml`：

1. **1:1 语法转写表与实测修正契约**：
   - **`V → .nod.xml`（含显式节点类型映射与闭合轮廓）**：
     - 每个有效节点 `v_p` 写出 `<node id="v_p" x="..." y="..." type="..." shape="..."/>`；
     - **`node_type → SUMO type` 确定性映射**（`configs/sumo_export.yaml`）：`junction_core` / `split_merge` / `midblock_node` / `road_continuation` 均映射为交付网的 `type="priority"`（SUMO 不接受自定义节点语义枚举名），`roi_entry_port` / `roi_exit_port` 映射为 `type="dead_end" fringe="outer"`；
     - **全内部节点显式 `shape` 契约（解决 `netconvert` 端点裁切漂移）**：经 SUMO 1.28.0 实测，即使开启 `--plain.extend-edge-shape=false`，若节点未写 `shape` 或车道端点偏离节点 `shape` 边界，`netconvert` 仍会自动裁切或延长自定义 `<lane shape>` 端点至其重算的路口轮廓边界。因此：`junction_core` 写出模型预测的 `v_p.polygon`（训练期受轮廓-端点一致性损失约束）；`split_merge` / `midblock_node` / `road_continuation` 由序列化器直接从其接入车道的模型预测外轮廓端点按角序确定性装配闭合 `shape`（纯 I/O 几何装配，零裁决启发式，SUMO 1.28.0 端到端实测端点漂移为 `0.000 m`）；
     - 顶层写入 `<location netOffset="-X_0,-Y_0" projParameter="+proj=utm +zone=49 +datum=WGS84 +units=m +no_defs"/>`；
   - **`(E_adj 链, L) → .edg.xml`（含显式 `<edge shape>`、默认速度/优先级与实测禁变道语法）**：
     - 每条链写出 `<edge id="e_k" from="v_u" to="v_v" numLanes="..." speed="13.89" priority="1" spreadType="right" shape="..."/>`：
       - **显式 `<edge shape>`（必需）**：按 `spreadType="right"` 规范，由该边最左侧机动车道（链尾 `lane_{k-1}`）的中心线向左法向确定性偏移 `w_{k-1}/2` 生成（不留空，避免 `netconvert` 回退为节点间直线再反向裁切车道端点）；
       - **默认车速与优先级（`configs/sumo_export.yaml`）**：因广州全库导航限速字段普遍为空，统一写出城市道路默认值 `speed="13.89"`（`50 km/h`）与统一 `priority="1"`（不做手写道路等级规则猜想；下游应用可按需覆写限速与路权）；
     - 按链序逐条写出 `<lane index="..." width="..." shape="..." .../>`；权限属性严格遵守 SUMO `allow` / `disallow` 互斥规范——`permission == bus_only` **仅写 `allow="bus"`（不写 `disallow`）**，其余机动车道（`general` / `variable` / `reversible`）写 `disallow="pedestrian"`（`variable` / `reversible` 另附 `<param key="lane_class" value="..."/>` 保留元数据）；
     - **实线禁变道属性实测契约**：经 SUMO 1.28.0 实测确认，`changeLeft="none"` 与 `changeLeft=""` 会导致 `netconvert` 直接报硬错退出（`Unknown vehicle class 'none'`），而 `changeLeft="ignoring"` 虽可编译但会被 `netconvert` 视为无限制默认值静默剥离（仿真中民用车辆仍可自由压实线超车！）。因此 `lane_change_flags` 为 `0` 时统一写出 **`changeLeft="authority"` / `changeRight="authority"`**（已在 `netconvert + sumo 1.28.0` 实测编译保留且对所有民用机动车辆严格生效拦截变道）；
   - **人行横道、辅助人行道与岛侧行人锚点机制（经 SUMO 1.28.0 实测验证，覆盖单向渠化右转与二次过街）**：
     > 实测证实：(1) SUMO 1.28.0 中若被跨越的边不含行人通道，`netconvert` 会报警 `Invalid pedestrian topology: crossing has no target` 并静默丢弃 `<crossing>`；(2) 若启用 `--sidewalks.guess` 或 `sidewalkWidth`，`netconvert` 会隐式插入 `lane index="0"` 造成机动车道索引 `+1` 错位；(3) 在**单向渠化右转道**或**中央安全岛二次过街**场景中，仅在被跨越边右侧加 `index="0"` 仍会因岛侧缺少步行区落脚点而被静默丢弃，且在同一边左侧追加第二条行人道会被 `netconvert` 报 `Ignoring additional sidewalk lane` 忽略；(4) `<crossing ... shape="..."/>` 为 SUMO 1.28.0 有效自定义几何属性名（`customShape` 会被忽略）。
     - **确定性处理（`AGENTS.md` 3.3 节允许的编译器语法接口例外）**：
       1. **边外侧辅助人行道**：凡被至少一条人行横道 `c_m ∈ C` 跨越的 `<edge>`，序列化器显式在其最外侧写入固定辅助人行道 `<lane index="0" width="1.5" allow="pedestrian" shape="..."><param key="aux_crossing_anchor" value="true"/></lane>`（由最外侧机动车道向右法向平移 `(w_0/2 + 0.75) m` 确定性生成），该边机动车道索引从 `1` 起严格递增，`.con.xml` 中的 `fromLane` / `toLane` 同步引用精确偏移后索引；
       2. **岛侧辅助行人锚点边（支持单向渠化右转与二次过街）**：当某条人行横道 `c_m` 跨越单向边或分隔带单侧车行道、其对侧（安全岛/中央分隔带侧）在所属节点上不存在既有步行区落脚点时，序列化器在该节点朝岛侧方向确定性挂载一条短辅助纯行人锚点边（`allow="pedestrian"`, `numLanes="1"`, `width="1.5"`, `spreadType="center"`, 长度 `2–4 m`，末端锚点节点 `type="dead_end"`，边与锚点节点均标注 `<param key="aux_crossing_anchor" value="true"/>`），已在 SUMO 1.28.0 上实测单向渠化右转与路段双侧二次过街均实现零告警、保形编译；
       3. 全部带有 `aux_crossing_anchor` 标记的辅助车道、锚点边与锚点节点**仅作满足 `netconvert` 拓扑语法要求的编译锚点（不参与道路拓扑裁决），在底图叠加可视化渲染、机动车 OD 仿真与机动车道几何指标评测中一律隐藏/排除**；
   - **`(E_succ, K_conn) → .con.xml`（含显式禁自动补连）**：
     - 每条激活连接写入 `<connection from="..." to="..." fromLane="..." toLane="..." shape="..."/>`（使用 `K_conn` 的 8 点无压岛内部轨迹）；
     - **禁止 `netconvert` 擅自补连（实测修正）**：经 SUMO 1.28.0 实测，若某条非边界出口边在 `.con.xml` 中未出现任何 `<connection>`，`netconvert` 会静默自动猜测并补出额外转向连接，破坏 $G$ 与 `net.xml` 的 1:1 映射。因此对 `E_succ` 中出度为 0 的非 `roi_exit_port` 边 `e_k`，序列化器**强制显式写出 `<connection from="e_k"/>`（不含 `to` 属性）**，经实测可 100% 禁止 `netconvert` 自动补连；
   - **`(C, E_cross, E_cross_node) → .con.xml`**：每条斑马线写入 `<crossing node="..." edges="..." width="..." shape="..."/>`（实测确认属性名为 `shape`）。
2. **`netconvert` 保真参数清单与强制编译后往返核验**：
   - **标准交付编译（不生成 `.tll.xml`，产出无信号基础路网 `site.net.xml`）**：
     ```bash
     netconvert \
       -n site.nod.xml -e site.edg.xml -x site.con.xml \
       --no-turnarounds=true \
       --junctions.join=false \
       --geometry.remove=false \
       --offset.disable-normalization=true \
       --plain.extend-edge-shape=false \
       --junctions.corner-detail=0 \
       -o site.net.xml
     ```
     *说明：`--junctions.join`、`--geometry.remove`、`--plain.extend-edge-shape` 在 SUMO 1.28.0 中默认即为 `false`，此处显式传入以锁定跨版本确定性；严禁开启 `--sidewalks.guess`、`--crossings.guess`、`--geometry.remove=true` 或 `--junctions.join=true`。*
   - **仿真测试台临时编译副本（仅用于 RL 奖励与饱和冲突流评测，输出临时 `eval_sim.net.xml`）**：
     - 经 SUMO 1.28.0 实测，`--tls.guess=true --tls.guess.threshold=0` 会错误地把带有斑马线的 `midblock_node` 也转为信号控制路口。因此临时测试台副本改用 **`--tls.set=<逗号分隔的全部 junction_core 节点 ID>`**，仅对模型显式预测的 `junction_core` 节点自动生成防死锁测试台信号配时（不触碰 `split_merge`、`midblock_node` 与 `road_continuation`），且绝不替代交付版 `site.net.xml`。
   - **强制编译后往返一致性核验（Post-Compile Round-Trip Verification，内置于 `src/sumo_io/`）**：
     每次 `netconvert` 编译后必须自动解析生成的 `*.net.xml` 并与输入图 $G$ 执行三项刚性比对，**任一项不通过即判编译失败**：
     1. **连接拓扑 100% 往返一致**：编译后非内部 `<connection>` 集合 `(from, to, fromLane, toLane)` 与 $G$ 的 `E_succ`（扣除辅助人行道索引偏移后）完全一致，凡出现 `netconvert` 自动补连或静默丢连一律判失败；
     2. **死胡同与无源车道独立核查**：鉴于 SUMO 1.28.0 实测证实 `netconvert` 不会对死胡同进口车道输出任何告警，死胡同/无源车道与被丢弃斑马线数一律直接基于 $G$ 与解析后的 `net.xml` 精确统计，不以 `netconvert` 零告警替代拓扑校验；
     3. **编译前后几何保真核查**：计算输入 $G$ 的路段车道中心线、`K_conn` 内部轨迹与编译后 `net.xml` 对应 `<lane shape>` / 内部车道 `<lane shape>` 的 Hausdorff 距离与端点漂移（M0 基准门禁 `≤ 0.15 m`，M1–M3 门禁 `≤ 0.20 m`）。

---

## 5. 零人工标注的三层学习闭环

### Layer 1：参数化合成引擎 Sim2Real 预训练（解决监督冷启动）

**核心洞察**：从碎片百度漆面正向猜 SUMO 拓扑极难，但从规范化 SUMO 有向车道图反向渲染百度风格漆面矢量精度极高且可无限随机化。

1. **规范化真值分段协议（Canonical Graph Ground-Truth Specification）**：
   为保证训练期 Hungarian 匹配目标唯一无歧义且全部节点均在底图上视觉可观测，合成引擎严格按统一协议生成真值图 $G^*$：
   - **节点唯一放置规则（严格取消固定弧长伪切分）**：仅在以下 **5 类视觉可观测事件**处创建节点并拆分车道段：(a) ROI 边界交点（`roi_entry_port` / `roi_exit_port`）；(b) 交叉口核心区停止线断面（`junction_core`，`pos` 为接入停止线端点几何质心，`polygon` 为精确经过所有接入车道外横断面角点并包络内部连接的规范 16 点闭合轮廓）；(c) 车道数或宽窄变化渐变段（`split_merge`）：对展宽/缩减渐变段 `[s_start, s_end]`，**上游 `k` 条恒宽车道段统一在渐变起点 `s_start` 终止，下游 `k±1` 条恒宽车道段统一在渐变终点 `s_end` 起始**（即展宽新增车道严格从达到完整标称宽度的 `s_end` 断面起存在，缩减消失车道严格在开始收窄前的 `s_start` 断面终止）；渐变段内部不放置任何 `L` 路段车道，仅在渐变段几何质心创建**单个 `split_merge` 节点**，并由 `K_conn` 的 8 点平滑分合流过渡轨迹桥接覆盖 `[s_start, s_end]`，从而严格保证全部 `L` 均为标量恒宽、Hungarian 匹配真值唯一无歧义；(d) 车道属性变化断面（如公交/可变/潮汐专用道起止、实虚分隔线禁变道切换，创建 `road_continuation` 节点切分）；(e) 路段功能事件断面（`midblock_node`）：在**路段中间掉头开口**（广州常见提前掉头，切分双向车道并生成跨 `midblock_node` 的 `E_succ` 与 `K_conn` 掉头轨迹）以及**非路口核心区人行横道中心断面**（如渠化右转道斑马线、路段中段人行横道，切分所跨机动车道并将 `E_cross_node` 绑定至该 `midblock_node`）处创建 `midblock_node`。每条路段车道均以 **`K_lane = 32`** 个等弧长控制点表示，不再因固定弧长插入依赖裁切框起点的伪切分节点；
   - **独立运动学内部轨迹生成器（解决 `netconvert` 监督矛盾）**：合成真值的连接过渡轨迹 $K_{\text{conn}}^*$（含跨 `junction_core` 路口内转向、跨 `split_merge` 渐变段分合流过渡以及跨 `midblock_node` 路段掉头；左转待转区走廊几何同步吸收进对应左转 $K_{\text{conn}}^*$）**不采用 `netconvert` 默认生成的内部曲线**，而是由参数化运动学样条生成器（$C^2$ 连续五次埃尔米特 / 回旋样条，端点位置与切向严格锁定进出口断面，显式满足安全岛/导流区/斑马线避障间隙、最小转弯半径约束与多车道并列转弯保序间隔）直接生成黄金轨迹并写入 `.con.xml`，经 4.3 节 `netconvert` 保真编译与**往返一致性核验**（连接集合 100% 一致、几何 Hausdorff 漂移 `≤ 0.15 m`、零告警）。
2. **场景多样性与显式负样本**：
   - **拓扑与几何多样性**：3–6 岔随机交角（`30°–150°`）、T 型/Y 型/十字/错位口/多岔口；直线与贝塞尔弯道引道；每臂独立随机进出口车道数（1–6 车道）与非对称配对布置；上游/进口展宽段与下游缩减段；中央双黄线/隔离带与框内掉头开口；转角安全岛 + 导流区 + 独立渠化右转道；人行横道（含中央安全岛二次过街与渠化右转道斑马线）；宽窄变异（`2.1 m–4.0 m`）；**专门合成广州高频场景：左转待转区（双白虚线 + 前端横白实线 + 内部左转箭头，输出端吸收进左转 `K_conn`）、路口内引导虚线、公交专用道、可变/潮汐车道**；
   - **显式负样本（支路与非机动过滤的纯数据驱动来源）**：在主干交叉口及引道周边随机挂接**无标线细小支路、园区出入口、地上停车背景面、停车位面、`parking_bidirectional_symbol` / `parking_oneway_symbol`、并入 `road_surface` 且由白实线或 `separation_structure_path` 分隔并带 `bicycle_lane_symbol` / `greenway_boundary` 的非机动车道，以及 `road_surface` 边缘带 `公交站点位` 的港湾式公交站**——全部渲染进输入 BEV 与矢量 Token，但在真值图 $G^*$ 与有效机动车道掩码 $M_{\text{valid}}^*$ 中**全部置空**；
   - **百度风格域随机化**：将车道边界按真实实测漆宽（`0.12–0.24 m`）渲染为带随机瓦片切缝的碎片 `MULTIPOLYGON`（白虚线独立漆段、白实线、横向停止线、黄实线/双黄线、导流斜纹、斑马线白条、组合导向箭头）；随机注入虚线长度波动、漆面磨损缺损、偶发箭头缺失、小幅几何抖动以及任意长宽比/旋转/偏移的 `roi_mask` 裁切（含跨分块裁切以监督 `port_embed` 跨块匹配）。
3. **存储与在线光栅化方案**：
   - **禁止离线缓存 36 通道 BEV 栅格**（`800×800×36×fp16 ≈ 46 MB/口`，5 万口需 ~2.3 TB，不可行）；
   - **明确采用「离线存轻量矢量与真值图 + DataLoader 在线选择性抗锯齿光栅化」**：生成 **30,000–50,000** 个通过 `netconvert` 往返一致性编译校验的合成交叉口，离线仅缓存紧凑矢量几何与真值图 $G^*$（每口 `~15–30 KB`，5 万口总占 **`< 1.5 GB`** APFS SSD）；训练时由多进程 DataLoader 在线执行随机 `roi_mask` 裁切、域随机化与 3.2 节选择性抗锯齿光栅化（窄标线 `4×4` 或解析亚像素线段覆盖 + 宽面 `1×` 精确光栅化）。
4. **预训练多任务损失**：
   $$\mathcal{L}_{\text{sim}} = \mathcal{L}_{\text{dense}} + \lambda_V (\mathcal{L}_{\text{node}} + \mathcal{L}_{\text{poly\_end}}) + \lambda_L (\mathcal{L}_{\text{pts}} + \mathcal{L}_{\text{dir}} + \mathcal{L}_{\text{width}} + \mathcal{L}_{\text{cls/perm/lc}}) + \lambda_K \mathcal{L}_{\text{conn\_traj}} + \lambda_C \mathcal{L}_{\text{cw}} + \lambda_{\text{topo}} \mathcal{L}_{\text{topo}}(E_{\text{adj}}, E_{\text{from}}, E_{\text{to}}, E_{\text{succ}}, E_{\text{cross}}, E_{\text{cross\_node}}) + \lambda_{\text{port}} \mathcal{L}_{\text{port\_match}}$$
   其中 $\mathcal{L}_{\text{poly\_end}}$ 约束 `junction_core` 闭合轮廓经过接入车道端点，$\mathcal{L}_{\text{port\_match}}$ 通过跨重叠窗端口对比损失监督 `port_embed` 用于 Phase 1/3 超大选区拼接。

### Layer 2：真实广州数据的物理自监督迁移（消除 Sim2Real 域差）

1. **零启发式真实交叉口自动采样（数据加载基础设施）**：
   - 从 55.54 GB 广州全库按地理空间分块隔离（见 7.1 节）自动采样 **3,000–6,000** 个真实平面交叉口（首先排除 `立交上下层关系` 表中存在上下层重叠的高架立交区域，留待 Phase 2）：
     1. **主采样源（原生设施点聚合）**：直接对 GPKG 原生图层 `信号灯点位`（全库 `8,545` 个点）及 `crosswalk_area`（`46,247` 个面）做空间聚类定位规范平交路口中心；
     2. **补充采样源（覆盖无信号灯规范路口）**：使用 Layer 1 预训练后的稠密 `junction_center` 预测头在含标线瓦片上滑窗扫描高置信路口中心（纯模型驱动，不依赖漏路的 `道路参考线` 或手工度数规则）；
   - 每个采样路口仅离线缓存轻量局部矢量及紧凑压缩的预计算可微几何场（约 `2.5–3.0 MB/口`，5,000 口总占 **`~12–15 GB`** APFS SSD）。
2. **预计算可微几何场（零启发式阈值定义）**：
   - **`SDF_boundary`（可贴边边界距离场）**：全部白/黄标线 + 道路面外边界 `∂road_surface`；
   - **`SDF_forbidden`（绝对禁入距离场）**：安全岛 `traffic_island`、导流区 `channelization_area`、隔离设施 `separation_structure_path`、道路面外部 `¬road_surface`，**以及显式并入的不可跨越实线场 `SDF_solid`**（`white_solid_stroke`、`white_double_solid`、`yellow_solid_stroke` 及混合线实线侧；配合可微切向方向门控 $|\cos(\theta_{\text{lane}} - \phi_{\text{marking}})|$ 与主干段 `[0.05, 0.95]` 权重，仅强惩罚纵向/斜向压实线，避免在进口端横向停止线处误罚！）；
   - **`SDF_divider`（车道分隔标线场）**：包含全部车道分隔线（白虚线 + 各类白/黄实线与混合线，同样按切向平行度加权排除横向停止线）；
   - **`M_zebra`（斑马线白条纹覆盖场）**：来自 `斑马线白色条纹`；
   - **`M_neg_explicit`（原生显式负语义区掩码）**：直接来自百度 GPKG 原生负类（各类停车位面、停车背景面、停车场内部通道符号、非机动与慢行区），**绝不使用任何手写“距标线超过 X 米”的阈值规则**。
3. **堵死全部退化解的物理自监督损失 $\mathcal{L}_{\text{phys}}$**：
   - **标线居中与边界贴合损失 $\mathcal{L}_{\text{align}}$**：对预测车道段每个控制点 `p_k` 与法向 `n_k`，左右边缘点 `p_k ± (w/2)·n_k` 最小化在 `SDF_boundary` 上的距离（仅在有标线/路缘支撑区域生效），并约束两侧对称，使中心线严格居中；
   - **禁入区零入侵损失 $\mathcal{L}_{\text{forbid}}$**：对车道包络采样点施加 $\text{ReLU}(m_{\text{forbid}} - \text{SDF}_{\text{forbidden}})$ 强惩罚，严禁压纵向实线、安全岛、导流区、护栏或冲出路面；
   - **反合并损失 $\mathcal{L}_{\text{no\_split}}$（堵死“两道合一宽车道”退化）**：在每条预测车道横向内部区间（相对中心线 `α ∈ [-0.35w, +0.35w]`，避开左右边缘标线）均匀采样内部网格点，最小化 $\text{ReLU}(m_{\text{div}} - \text{SDF}_{\text{divider}})$，彻底禁止车道内部吞并白虚线或实线；
   - **防漏检覆盖与防重叠损失 $\mathcal{L}_{\text{cov}} + \mathcal{L}_{\text{overlap}}$（堵死“少建/不建车道”退化）**：
     - **箭头必覆盖损失**：对 ROI 内每个真实 `arrow_*` 实体，要求至少被一条高置信度进口车道软包络覆盖（$-\log \max_i P_{\text{cover}}(i, a)$）；
     - **有效标线机动车道覆盖 Soft-Dice（扣除路口核心区、斑马线区与非机动渗漏防线）**：为防止 EMA 教师整体漏检导致学生自我确认坍塌，同时**避免将路段车道 `L` 推过停止线侵入斑马线与路口核心区，且避免因实线外侧邻近非机动车道而误建机动车道**（见 3.2 节实测：`bicycle_lane_symbol` 位于 `road_surface` 内且距外侧实线仅 `0.77 m`），先定义非核心区有效路段域 $M_{\text{road\_seg}} = \text{road\_surface} \setminus (M_{\text{neg\_explicit}} \cup M_{\text{crosswalk}} \cup \text{sg}(M_{\text{junc\_core}}^{\text{EMA}}))$；在该域内，**客观防坍塌下限 $M_{\text{motor\_support}}$ 严禁对全部实线做各向同性膨胀**，而仅由**机动车专属正向证据**构成——即全量 `arrow_*` 覆盖域与机动车内部分道/专用标线（`white_dashed_stroke`、`white_dashed`、`bus_lane_marking`、`variable_lane_marking`、`reversible_lane_marking`）的距离衰减软核 $\exp(-\text{UDF}_{\text{motor\_marking}} / \sigma_{\text{cov}})$，并显式扣除 `M_neg_explicit`；无内虚线的单车道机动路段由箭头证据及经 Layer 1 非机动/港湾公交站负样本训练的 EMA 教师 $M_{\text{EMA}}$ 共同承担，取带客观下限的软融合 $M_{\text{target}} = \max(\alpha_{\text{floor}} \cdot M_{\text{motor\_support}},\; M_{\text{EMA}} \cdot M_{\text{road\_seg}})$；预测侧将路段车道软包络 $\hat{M}_{\text{lane}}$ 与 `K_conn` 的可微车体扫掠包络 $\hat{M}_{\text{conn}}$ 取并集 $\hat{M}_{\text{cov}} = \max(\hat{M}_{\text{lane}}, \hat{M}_{\text{conn}})$（自然覆盖 `split_merge` 渐变过渡段与 `midblock_node` 掉头开口），驱动 $\hat{M}_{\text{cov}} \cdot M_{\text{road\_seg}}$ 对 $M_{\text{target}}$ 最大化 Soft-Dice；
     - **车道间横向防重叠损失**：惩罚不同路段车道软包络的重叠面积，禁止用重复堆叠车道刷覆盖率；
   - **双向箭头方向与转向精确率/召回率拓扑损失 $\mathcal{L}_{\text{arrow}}$（堵死“全连接刷召回”漏洞）**：
     - 覆盖箭头的进口车道切向与箭头朝向余弦趋近 1；
     - **可微软转向分类定义**（角度中心、带宽与容忍权重统一置于 `configs/losses.yaml`）：对任意跨节点候选对 `(l_i, l_j)`，计算进口终点切向 $\mathbf{t}_i$ 与出口起点切向 $\mathbf{t}_j$ 的有符号偏转角 $\Delta\psi_{ij} = \text{atan2}(\mathbf{t}_i \times \mathbf{t}_j,\; \mathbf{t}_i \cdot \mathbf{t}_j) \in (-\pi, \pi]$（辅以起终点弦向方位判别掉头与错位口），经平滑角度核（von-Mises / 高斯 softmax，中心对应 `[L, S, R, U, FL, FR]` 六类标准偏转角）映射为可微转向概率分布 $q_{ij} \in \Delta^6$；
     - 经 noisy-OR 聚合得到进口车道 $i$ 在 6 类转向上的预测激活概率 $\hat{y}_{i, c} = 1 - \prod_j (1 - E_{\text{succ}}[i, j] \cdot q_{ij, c})$，对其匹配到的地面箭头并集多热标签 $y_i \in \{0, 1\}^6$ 施加**双向加权二元交叉熵（Recall + Precision BCE）**：$y_{i,c}=1$ 项强制覆盖全部指示转向，$y_{i,c}=0$ 项显式扣罚箭头未指示的多余转向连接（对邻近斜角 `L/FL`、`R/FR` 在 `configs/losses.yaml` 中设置温和混淆容忍权重，对明确冲突转向如纯直行连左转/掉头施加强惩罚）；
   - **路口内部连接轨迹物理损失与车道级配对代理监督 $\mathcal{L}_{\text{conn\_phys}} + \mathcal{L}_{\text{pair\_proxy}}$**：
     > **域差透明声明**：真实广州数据无路口内“进口车道 $i$ → 出口车道 $j$”的车道级真值标注，$\mathcal{L}_{\text{arrow}}$ 仅能监督至出口臂（转向类别）粒度。为缓解路口内车道级配对过度依赖合成先验，Layer 2 额外引入三项可微物理/底图代理监督，并由 Layer 3 SUMO 闭环车体包络与碰撞奖励兜底消除剩余非对称配对域差。
     - 对 `K_conn` 施加五重约束：(a) 与前后车道段端点 $C^1$ 位置+切向连续；(b) 沿轨迹 `2.2 m` 标准车体宽包络对安全岛/导流区/路外零入侵；(c) 曲率平滑与物理最小转弯半径上限约束（普通转向 $R_{\min} \ge 6.0\text{ m}$，掉头 $R_{\min} \ge 4.5\text{ m}$）；(d) 同进口并列多条转向连接的横向保序与防交叉安全间隔损失；(e) **方向门控路口内引导虚线与左转待转区贴合约束**：对路口核心区内与 `K_conn` 局部切向近平行（$|\cos(\theta_{\text{conn}} - \phi_{\text{marking}})|$ 高）的 `white_dashed_stroke` 引导线施加软分隔防穿越损失，直接利用底图真实引导虚线约束并列左转/直行轨迹分道；
     - **车道级配对可微代理损失 $\mathcal{L}_{\text{pair\_proxy}}$**：(1) **可行性期望耦合**：将 $E_{\text{succ}}[i,j]$ 与停梯度后的 `K_conn(i,j)` 曲率/横向偏移/避障代价值相乘求期望，直接压低跨道急弯扭曲连接概率；(2) **同臂单调不交叉软惩罚**：对同 `(edge_in, edge_out)` 内满足 $i_1 < i_2, j_1 > j_2$ 的交叉对施加 $\sum E_{\text{succ}}[i_1, j_1] E_{\text{succ}}[i_2, j_2]$ 惩罚；(3) **出口车道软覆盖约束**：对同节点下每条活跃出口车道 $j$ 施加 $\text{ReLU}(1 - \sum_i E_{\text{succ}}[i, j])$ 惩罚，防止同一出口臂的多个进口连接全部塌缩至单条出口车道；
   - **路口轮廓-端点一致性与斑马线白条贴合损失 $\mathcal{L}_{\text{poly\_end}} + \mathcal{L}_{\text{cw\_phys}}$**：约束 `junction_core` 的 `polygon` 边界贴合其接入车道起止横断面角点（保障编译几何零漂移），并最大化预测人行横道矩形区域与 `M_zebra` 的 Soft-Dice；
   - **负语义抑制与 Sim+Real 混合训练**：在 `M_neg_explicit` 区域压低存在置信度，并按比例混合 Layer 1 合成数据（含无标线支路、非机动车道与港湾式公交站负样本）联合训练，既保持完整节点/拓扑语法，又以纯学习方式持续抑制非目标道路。
4. **防选择偏差的渐进局部自训练与难例挖掘**：
   - 为避免"只有简单路口通过门禁，导致难例域差固化"，采用两项机制：(a) **局部子图伪标签（Partial Pseudo-Labeling）**：在难例路口中，对通过单臂/单连接几何与箭头核验的局部实体构建带置信掩码的部分伪标签，其余争议区域维持 $\mathcal{L}_{\text{phys}}$ 约束；(b) **渐进课程门禁 + 难例重采样**：结合 Mean-Teacher EMA 渐进收紧门禁，按 $\mathcal{L}_{\text{phys}}$ 残差分位数提高多岔/斜交/大展宽难例的采样权重，并将难例几何形态分布反哺至 Layer 1 合成生成器参数采样分布（**严格限定仅统计 `train` 集难例，严禁使用 `val` / `test` 集统计信息以杜绝任何数据泄漏**）。

### Layer 3：SUMO-in-the-Loop 强化学习精修（优化不可微仿真目标）

前两层无法对离散的 `netconvert` 编译校验与 `sumo` 交通微仿真动力学直接求导，因此第三层引入仿真闭环强化学习，并针对高维动作空间与奖励漏洞做彻底加固：

#### 1. 确定性无盲区仿真测试台协议

- **临时测试台配时（精确限定至 `junction_core`）**：仅对 RL/评测临时编译副本 `eval_sim.net.xml` 启用 **`netconvert --tls.set=<junction_core_ids>`**（取代会误给路段 `midblock_node` 建灯的 `--tls.guess --tls.guess.threshold=0`），剥离无信号优先权路口全转向死锁引发的假性 teleport，防止模型为躲避死锁而删减合法转向；
- **强制路口内碰撞检测与多种子确定性仿真**：`sumo` 运行强制开启 `--collision.check-junctions=true --collision.action=warn`，采用「稀疏全可达 OD 连通探测流（确保每条进口车道到每个合法转向均有车跑通）+ 中等稳态冲突车流」双档仿真，使 `collisions == 0`、`teleports == 0` 成为真实、无盲区、低噪声的确定性信号。为防止奖励与评测对单一随机车流过拟合，**训练期**在同一路口同轮 `K` 个候选间共享相同固定种子（消除候选间种子噪声），但在不同迭代轮次间轮换固定多种子池（如 `{42, 137, 2026}`）；**验证与最终评测期（`val / test`）**统一使用 **3 个固定随机种子**（`--seed ∈ {42, 137, 2026}`）取均值报告。

#### 2. 反刷分复合奖励函数 $R(G)$（含实体级局部归因）

$$R(G) = w_1 R_{\text{align}} + w_2 R_{\text{no\_intrusion}} + w_3 R_{\text{recall}} + w_4 R_{\text{no\_merge}} + w_5 R_{\text{filter}} + w_6 R_{\text{topo}} + w_7 R_{\text{sumo}}$$

- $R_{\text{align}}$：路段中心线相对标线中轴横向偏移（P50 / P90）与斑马线白条 IoU；
- $R_{\text{no\_intrusion}}$：纵向压实线长度、安全岛/导流区/护栏侵入面积的硬惩罚；
- **$R_{\text{recall}}$（堵死少建车道刷分，且与非机动排除域一致）**：非核心区/非斑马线/排除 `M_neg_explicit` 的有效标线机动车道软覆盖 F1（计入 `L ∪ K_conn` 扫掠包络，与 Layer 2 的 $M_{\text{motor\_support}}$ 口径严格对齐）、导向箭头被进口车道覆盖率、以及有效进出臂完备度；
- **$R_{\text{no\_merge}}$（堵死宽车道合并刷分）**：车道内部横向主体区穿越分隔标线（虚线/实线）的强负惩罚；
- $R_{\text{filter}}$：在原生显式负语义区（停车区、非机动车道、`M_neg_explicit`）与低支撑无标线区误建车道的负奖励；
- $R_{\text{topo}}$：导向箭头转向精确率与召回率（F1）、跨臂转向交叉冲突惩罚，以及**约束解码前原始输出与合法解码图的一致性奖励**（即使阶段 E 兜底消除死胡同/同臂交叉，仍对原始 logits 的非法倾向扣分）；
- $R_{\text{sumo}}$：`netconvert` 零错编译 + **100% 往返连接一致（零自动补连/丢连）+ 几何 Hausdorff 漂移达标**的前提奖励 + 全 OD 到达率 + `--collision.check-junctions` 下 `collisions == 0` / `teleports == 0` + **路口内部转弯车辆车体包络（Swept Volume）防侵入安全岛/导流区与平滑度奖励**（仅计路口内部动态包络，避免与路段 $R_{\text{align}}$ 重复计分）。

#### 3. 动作空间解耦、可分解近似似然与稳态优化路线（解决高维 GRPO 梯度崩塌）

针对“全连续控制点 + 独立伯努利边采样导致 95%+ 非法图与极弱信用分配”的核心风险，采用**解耦动作空间 + 结构化合法采样 + 可分解近似对数似然 + 实体级归因 + 多对偏好 DPO 先行**：

1. **动作空间解耦**：连续控制点与宽度保留由 Layer 2 高信噪比可微物理损失联合保真，**Layer 3 冻结/锚定路段几何主干，聚焦于低维离散结构决策**——(a) 模糊置信带（`score ∈ [0.2, 0.8]`）节点/车道的去留；(b) `E_adj`、`E_from/E_to`、`E_succ`、`E_cross` 与 `E_cross_node` 的离散拓扑选择；
2. **结构化合法采样与可分解近似对数似然（Factorized Surrogate Log-Likelihood）**：
   - 严禁对拓扑矩阵独立伯努利采样！改为对边/节点 logits 注入受控 Gumbel 噪声后经过 **4.2 节阶段 E 的约束图解码器**生成候选图，强制满足每条车道最多 1 个左邻、无环、同边共享节点、`E_succ` 节点一致/内部非零度/同臂保序无交叉等结构公理，将候选图 `netconvert` 往返一致合法率提升至 `> 85%–95%`，彻底杜绝整组奖励坍缩；
   - **近似对数概率显式定义（非精确整图似然）**：由于组合优化约束解码器隐含不可解的全局合法图配分函数 $Z(\theta)$，经 Gumbel + 约束解码得到的候选图无法闭式求解精确整图对数概率 $\log \pi_\theta(G)$。因此，DPO 与 GRPO 显式采用解码器前端头输出的**可分解伪似然近似（Factorized Surrogate Log-Likelihood）**替代精确联合似然：记 $a(G)$ 为候选图对应的离散决策向量（模糊置信带节点/车道存在标志及 `E_adj`、`E_from`、`E_to`、`E_succ`、`E_cross`、`E_cross_node` 的离散取值），对 DPO 偏好对 $(G_{\text{win}}, G_{\text{lose}})$ 仅在**两图存在差异的边与节点子集** $\Delta(G_{\text{win}}, G_{\text{lose}})$ 上对逐项对数概率求和：
     $$\log \frac{\tilde{\pi}_\theta(G_{\text{win}})}{\tilde{\pi}_{\text{ref}}(G_{\text{win}})} - \log \frac{\tilde{\pi}_\theta(G_{\text{lose}})}{\tilde{\pi}_{\text{ref}}(G_{\text{lose}})} \triangleq \sum_{k \in \Delta(G_{\text{win}}, G_{\text{lose}})} \left( \log \frac{p_\theta(a_k^{\text{win}} \mid x)}{p_{\text{ref}}(a_k^{\text{win}} \mid x)} - \log \frac{p_\theta(a_k^{\text{lose}} \mid x)}{p_{\text{ref}}(a_k^{\text{lose}} \mid x)} \right)$$
     相应地，GRPO 不计算整图联合重要性比率，而是对每个有效/差异决策变量 $k$ 直接采用因子化重要性比率 $r_k(\theta) = \frac{p_\theta(a_k \mid x)}{p_{\text{old}}(a_k \mid x)}$ 搭配裁剪目标与因子化 KL 锚定（明确声明其为忽略约束配分函数的可分解优化近似，而非精确概率密度）。
3. **方差归因前置检验**：进入 RL 更新前，先对验证集抽样做奖励方差归因检验（确认同路口不同结构化候选图间的奖励方差显著且主要由拓扑决策差异驱动，而非仿真随机噪声）；
4. **两阶段稳态 RL 落地与轮次抽样预算（方案 A 主力 + 方案 B 可选叠层）**：
   - **方案 A1（首选冷启动）— 分层抽样多对偏好迭代优化（Multi-Pair Iterative DPO）**：为匹配本机算力预算（避免每轮全量 3–6 千口 × 16–32 候选产生 5–19 万次仿真导致单轮耗时 2–7.5 h），**每轮迭代从真实训练集按难例优先 + 跨区分层抽样选取 `500–1,000` 个路口**，每口通过结构化采样生成 `K = 16` 个合法候选图并多进程并行评估复合奖励 $R(G)$（单轮约 `0.8–1.6 万` 次仿真，7 进程预计 `20–40 min/轮`，支持在预算内完成 `4–8` 轮迭代；具体抽样量与 `K` 值在 M0 实测吞吐后校准）；为避免“16 个候选仅取 1 对最高/最低分”导致 87.5% 仿真样本浪费，从同口全部有效候选中按奖励排序构造**满足显著奖励间隔 $R(G_a) - R(G_b) \ge \delta_R$ 的多组偏好对（如 Top-4 与 Bottom-4 配对，按奖励间隔归一化加权）**，基于上述差异子集 $\Delta(G_a, G_b)$ 的可分解近似对数似然执行带 KL 锚定的多对 DPO 微调，样本利用率提升 `4–8×` 且收敛稳定；
   - **方案 A2（在线精细探索）— 实体级局部优势结构化 GRPO**：在 DPO 收敛后的高合法率策略上，同样采用每轮 `500–1,000` 口难例抽样，将奖励分解为全局仿真项 + **按车道/连接归因的局部奖励**（如某条连接触发路口碰撞、压岛或未覆盖箭头时直接扣减对应 `(i, j)` 边优势 $A_k$），以因子化比率 $r_k(\theta)$ 与 KL 锚定执行结构化 GRPO 精修；
   - **方案 B（进阶可选叠层）— 独立 GNN 多步图精修智能体（PPO）**：仅当方案 A 在超复杂多岔口遇到明确长尾瓶颈时启用，以方案 A 输出为 $s_0$，在 `T = 5~8` 步内对争议边/节点执行多步局部精修。

> **训练/损失/奖励超参数与技术红线边界声明**：本节及 4.3 节中涉及的各项物理、优化与编译器导出常数（如主干段实线门控区间 `[0.05, 0.95]`、反合并内部采样带宽 `±0.35w`、覆盖软核参数 `σ_cov` 与下限 `α_floor`、可微转向角核中心/带宽/精确率容忍权重、内部轨迹标准车体包络宽 `2.2 m`、物理最小转弯半径 `R_min = 6.0 m / 4.5 m`、安全间隙 `m_forbid / m_div`、RL 探索模糊置信带 `[0.2, 0.8]`、多对 DPO 奖励间隔阈值 `δ_R`，以及 SUMO 默认限速/优先级/节点枚举映射）均属于 `AGENTS.md` 3.3 节允许的**损失函数、RL 奖励、训练策略数学超参数与确定性 I/O 配置**，统一收口于 `configs/` 配置文件管理与消融实验，**绝不作为推理期硬编码几何/拓扑裁决规则**。

---

## 6. 本机算力（Apple M4 Pro 24 GB）与工程落地方案

| 资源/约束项 | 规格与落地设计 |
| --- | --- |
| **计算硬件** | Apple M4 Pro（12 核 CPU：8P+4E，24 GB 统一内存，PyTorch 2.8.0 MPS 加速） |
| **模型体量与 MPS 算子适配** | 总参数量控制在 **25M–40M**（高保真轻 Stem + `ConvNeXt-Tiny`/`ResNet34` + `F_2(D_2=128)` + 轻量矢量编码器/6 层 MapTRv2 解耦层级解码器/GNN）；**解耦 Intra/Inter-Instance 注意力**将 `K_lane=32` 下的解码器自注意力开销压低 `>20×`，支撑 `1024×1024` 单窗推理与 `512–896` 训练；Deformable Cross-Attention 采用 **纯 MPS 原生 `gather` 可微双线性采样** 替代 PyTorch 2.8.0 MPS 尚不支持反向的 `grid_sample`，杜绝 CPU fallback；训练采用 `batch_size = 4~8` + 梯度累积，并在 M0 验证 MPS `fp16/bf16` AMP 数值稳定性 |
| **性能实测门禁与时长预算** | **拒绝未经验证的吞吐臆测**：M0 阶段必须在本机实测三项基准：(1) **选择性抗锯齿光栅化（窄标线 `4×4` + 宽面 `1×`）vs 解析亚像素带宽线段覆盖率光栅化**的单图 CPU 耗时（若 `4×4` 方案单样本超 `0.25 s` 或 8 进程有效吞吐低于 GPU 喂食速率，立即切换至解析覆盖率方案，避免 125 万样本光栅化成为 40+ h 瓶颈）；(2) 单图前向/反向延迟与显存占用（含 `896×896` 训练与 `1024×1024` 推理）；(3) `6–8` 进程池并行执行 `netconvert + 往返核验 + sumo` 的实际路口/秒吞吐。初步训练时长预算（**须待 M0 实测吞吐后重新核算校准**）：Layer 1 合成预训练（3–5 万口，~20–30 epochs）约 `24–48 h`；Layer 2 真实物理自监督与渐进自训练约 `12–24 h`；Layer 3 采用每轮分层抽样 `500–1,000` 口 × `K = 16` 候选执行 `4–8` 轮多对偏好迭代 DPO（及可选 GRPO）约 `8–16 h` |
| **统一 Python 环境** | 在**本机 APFS 磁盘**维护独立环境（如 `~/.venvs/baidu2sumo` 或 conda `baidu2sumo`），统一安装 PyTorch 2.8（MPS）+ GeoPandas/Shapely + OpenCV + SciPy + `sumolib`/`traci`，并调用 `~/.microsim-platform/venv/bin/{netconvert,sumo}`（SUMO 1.28.0）；**严禁在 exFAT 建 venv，严禁使用 `/opt/homebrew/bin/python3`**，运行 Python 默认加 `-B` |
| **APFS 存储预算与 I/O 分离** | 原始 55.54 GB GPKG 严格只读；APFS 高速 SSD 缓存目录（如 `~/cache/baidu2sumo/`）总空间预算 **`~25–35 GB`**：合成轻量矢量与真值（5 万口 `< 1.5 GB`，在线光栅化）+ 真实广州切片与紧凑压缩 SDF 场（5 千口 `~12–15 GB`）+ 训练 checkpoints 与仿真日志（`~5–10 GB`）；exFAT 仓库内仅保存源码、配置、轻量化切分清单、指标报告与精选叠图 |

---

## 7. 评测体系与量化里程碑路线

### 7.1 防泄漏空间切分、双轨量化与视觉终审

1. **严格地理空间隔离与隔离带铁律**：
   - **`开创大道_WGS84.gpkg` 仅作开发冒烟与调参验证集（`dev / val-dev`），严禁进入最终 `test` 留出集**（且其所处空间区块整体排除出 `train` 与 `test`）；
   - 全广州 11 区按行政区 + `5 km × 5 km` 空间区块划分 `train / val / test`，并在不同集合相邻区块交界处强制预留 **`≥ 500 m` 地理隔离缓冲带**（远大于最大 `204.8 m + 32 m` 单窗含缓冲跨度），彻底消除任何跨界数据泄漏。
2. **双轨客观评测 + 强制底图视觉终审**：
   - **轨道 1 — 大规模真实自动评测（全部 `val / test` 留出路口，数百口）**：
     - 底图几何贴合：中心线相对标线中轴横向偏移（均值 / P90 / P95）、纵向压实线率、安全岛/导流区侵入面积比、车道内部分隔线穿越率（反合并指标）、非核心区/非斑马线/排除非机动有效标线机动车道覆盖 F1（计入 `L ∪ K_conn` 扫掠包络）、斑马线白条 IoU、原生负语义区与非机动车道误建率；
     - 拓扑与箭头一致性（**强制双轨统计：约束解码前原始输出 vs 阶段 E 约束解码后输出**）：导向箭头转向精确率、召回率与 **F1**（防多连刷召回）、死胡同车道率（raw / decoded）、无源车道率（raw / decoded）、同进口转向交叉冲突数；
     - SUMO 双网编译与闭环仿真健康度：**往返一致编译成功率**（`netconvert` 零错 + 连接集合 100% 往返一致零自动补连/零丢连 + 编译前后车道/连接几何 Hausdorff 漂移 `≤ 0.20 m`）；**交付版 `site.net.xml` 稀疏单 OD 无冲突全可达仿真到达率**；临时测试台副本 `eval_sim.net.xml`（仅 `junction_core` 启用 `--tls.set`）+ `--collision.check-junctions` 在 **3 个固定随机种子**（`{42, 137, 2026}`）下的全 OD 到达率均值、`collisions == 0` 通过率、`teleports == 0` 通过率、路口内车体包络无压岛率。
   - **轨道 2 — Eval-Only 人工核验审计集（20–40 个跨区代表性真实测试路口，只读评测，绝不参与训练与调参，设 M2/M3 刚性门槛）**：
     - 彻底打破自动指标与训练损失同源的“自参考刷分”盲区，独立核验**逐臂真实车道数精确率、真实转向矩阵 F1、真实无标线支路与非机动车道误建率（FPR）与人行横道召回率**（首版结合用户抽检结果固化为只读评测清单，并在 M2 / M3 里程碑设置明确量化门禁）。
   - **强制底图叠加视觉终审**：每次里程碑批量输出「百度高清底图 + 预测节点/路口轮廓/车道中心线/标量宽度边缘/斑马线/内部连接轨迹 + SUMO 仿真车辆轨迹快照」的全景图与核心放大图；**标量指标不得替代肉眼终审**，存在明显压岛、断连、反向、错道或支路/非机动车道残留即判不通过。

### 7.2 分阶段量化里程碑（Phase 1）

| 里程碑 | 核心交付 | 可机检量化验收门槛 + 视觉门禁 |
| --- | --- | --- |
| **M0：工程底座、数据契约与性能基准** | 统一 APFS ML+SUMO 环境；`EPSG:32649` 任意 ROI 提取 + 选择性抗锯齿 / 解析覆盖率 36ch 在线光栅化 + 矢量 Token 化；完整图数据结构 $G$（`K_lane=32`）、增强约束图解码器、超大选区确定性分块端口拼接与 SUMO 序列化器（含显式 `<edge/node shape>`、`changeLeft="authority"`、岛侧辅助行人锚点边、`<connection from="..."/>` 禁补连、`--tls.set` 与编译后往返核验）；`≥ 500 m` 隔离带跨区切分清单；本机 M4 Pro 算子/光栅化/仿真吞吐实测基准报告 | 1. 单元测试 100% 通过（含超大选区分块端口拼接单测）；<br>2. 手工构造的 20+ 种复杂拓扑标准图（须覆盖展宽、单向渠化右转斑马线、中央岛二次过街、路段掉头、实线禁变道、零出边禁自动补连、公交道）`netconvert` 零错零告警编译率 **100%**、**往返连接集合 100% 完全一致**、**编译前后车道/内部连接几何 Hausdorff 漂移 `≤ 0.15 m`**，`sumo`（含 `--collision.check-junctions`）`collisions == 0`、`teleports == 0` 且 `changeLeft="authority"` 实测阻断民用车变道；<br>3. MPS 原生双线性采样数值误差 `< 1e-5`（相对 CPU `grid_sample`）且反向零 CPU fallback；完成 `4×4` 选择性抗锯齿 vs 解析覆盖率吞吐基准落盘并定型 |
| **M1：合成引擎 + DL 基座（Layer 1）** | 5 类视觉可观测事件规范化分段协议（`K_lane=32`）+ 运动学内部轨迹生成器 + 3–5 万合成轻量矢量库（含左转待转区、路口引导虚线、公交/可变/潮汐车道、无标线支路/非机动车道/港湾式公交站负样本、跨块端口对比样本与任意 `roi_mask`）；双编码器融合 + MapTRv2 解耦层级解码器训练完成 | **合成留出测试集（含 `>172.8 m` 分块拼接验证）**：<br>• 车道中心线横向误差 `P50 ≤ 12 cm`、`P90 ≤ 25 cm`；<br>• 有效车道与人行横道 F1 `≥ 92%`，无标线支路与非机动车道误建率 `FPR ≤ 2%`；<br>• 导向箭头转向 F1 `≥ 92%`；<br>• **往返一致编译成功率 `≥ 95%`（`Hausdorff ≤ 0.20 m`）**，交付网稀疏单 OD 与测试台（`--tls.set`）全 OD 到达率 `≥ 92%`，`collisions == 0` 通过率 `≥ 95%`；<br>**真实验证集**：输出首批 Zero-shot 底图叠图与域差诊断报告 |
| **M2：真实物理自监督迁移（Layer 2）** | 3–6 千真实广州平交口零启发式自动采样与紧凑 SDF 场缓存；防退化完整 $\mathcal{L}_{\text{phys}}$（含反合并 $\mathcal{L}_{\text{no\_split}}$、防非机动渗漏的 $M_{\text{motor\_support}}$ 覆盖 $\mathcal{L}_{\text{cov}}$、双向箭头精确率/召回率 $\mathcal{L}_{\text{arrow}}$、路口内车道级配对代理损失 $\mathcal{L}_{\text{pair\_proxy}}$、轮廓-端点一致性与纵向实线门控禁入）；Sim+Real 混合训练 + 局部渐进自训练与 `train`-only 难例反哺 | **真实广州 `val` 留出集（含双轨 raw/decoded 统计）**：<br>• 车道中心线相对标线中轴横向偏移 `P50 ≤ 15 cm`、`P90 ≤ 35 cm`；<br>• 纵向压实线率 `≤ 3%`，安全岛/导流区侵入率 `≤ 1%`，**车道内部分隔线穿越率（反合并）`≤ 2%`**；<br>• **有效标线机动车道覆盖 F1 `≥ 88%`**，导向箭头覆盖率 `≥ 90%` / **转向 F1 `≥ 85%`**，斑马线白条 IoU `≥ 0.72`；<br>• 无标线支路/负语义区/非机动车道误建率 `FPR ≤ 5%`；<br>• **往返一致编译成功率 `≥ 92%`（`Hausdorff ≤ 0.20 m`）**；<br>• **Eval-Only 人工核验审计集初版基线**：逐臂车道数精确率 `≥ 82%`、转向矩阵 F1 `≥ 80%`；视觉终审无大面积偏移与明显支路/非机动残留 |
| **M3：SUMO-in-the-Loop RL 精修（Layer 3）** | 奖励方差归因检验；结构化合法采样 + 多种子确定性仿真测试台（`--tls.set=<junction_core_ids>` + `--collision.check-junctions`）+ 每轮分层抽样与可分解近似似然多对偏好迭代 DPO（及可选结构化 GRPO / 方案 B 叠层） | **跨区留出真实 `test` 集（排除开创大道，3 固定种子均值）**：<br>• 中心线横向偏移 `P50 ≤ 12 cm`、`P90 ≤ 28 cm`，斑马线白条 IoU `≥ 0.78`；<br>• 压实线率 `≤ 1.5%`，安全岛/导流区侵入率 `≤ 0.5%`，**车道内部分隔线穿越率（反合并）`≤ 1.0%`**，**有效标线机动车道覆盖 F1 `≥ 92%`**，无标线支路/非机动车道 `FPR ≤ 3%`；<br>• 导向箭头覆盖率 `≥ 95%` / **转向 F1 `≥ 92%`**；死胡同/无源车道率 **raw `≤ 3%`、decoded `≤ 0.5%`**；<br>• **往返一致编译成功率 `≥ 98%`（`Hausdorff ≤ 0.20 m`）**，**交付版 `site.net.xml` 稀疏单 OD 到达率 `≥ 96%`**，测试台 `eval_sim.net.xml` 全 OD 到达率 `≥ 96%`、`collisions == 0` 通过率 `≥ 98%`、`teleports == 0` 通过率 `≥ 95%`；<br>• **Eval-Only 人工核验审计集（20–40 口）硬性门禁**：逐臂真实车道数精确率 `≥ 90%`、真实转向矩阵 F1 `≥ 90%`、真实无标线支路与非机动车道 `FPR ≤ 3%`、人行横道召回率 `≥ 92%`；<br>• 批量交付底图叠加与仿真轨迹视觉终审包（含审计对照表）供用户实测 |
