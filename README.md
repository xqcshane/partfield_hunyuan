# Hunyuan3D → PartField → FitOBJ：Agent 自动选参完整说明（V32）

本 README 面向两类使用者：

1. 人工运行 `python -m partfield_mc.cli`；
2. 将本文交给 AI Agent，由 Agent 根据对象类型、目标风格和已有中间文件自动生成参数。

本文以 **V32** 实际 CLI 为准，覆盖：

- Hunyuan3D、PartField、FitOBJ 三阶段拆分运行；
- 全部 `--fit-mode`：`obb / aabb / shared / primitive`；
- 全部 `--obj-mode`：`merged / separate / surface / all`；
- Primitive 的部件分类、候选类型、接触模式、接触强度和减面策略；
- Agent 的确定性选参规则、禁止组合和完整命令模板；
- 苹果、狐狸、人物、方块模型等预设参数。

> **重要事实：V32 的 Primitive 候选由代码程序化生成。** 目前不会读取外部 `OBJ/GLB` 预设模型库。`box / prism / frustum / cone / ellipsoid / convex` 都是运行时生成和评分的候选。

---

# 1. 流水线与可复用阶段

```text
图片
  ↓ Stage 1：Hunyuan3D
原始高精度带纹理 GLB
  ↓ Stage 2：PartField
normalized_mesh.ply + labels.npy
  ↓ Stage 3：FitOBJ
Cuboid 或 Primitive Paper Model
```

三种运行范围：

| 已有输入 | 需要运行 | 说明 |
|---|---|---|
| 只有图片 | Stage 1 + 2 + 3 | 使用图片参数和 `--texture` |
| 已有带纹理 GLB | Stage 2 + 3 | 把 GLB 作为位置参数，不再调用 Hunyuan3D |
| 已有 GLB、PLY、NPY | 只运行 Stage 3 | 使用 `--postprocess-only` |

修改 FitOBJ 代码或参数时，只需要重复 Stage 3。

---

# 2. Agent 在生成命令前必须获取的信息

Agent 至少需要明确以下字段。缺少关键字段时，应先询问，不得猜测文件路径。

```yaml
input_state:
  has_images: true|false
  has_textured_glb: true|false
  has_normalized_mesh: true|false
  has_labels: true|false

object_profile:
  category: generic|animal|person
  shape_style: blocky|mostly_boxy|organic|mixed
  continuous_main_body: true|false
  thin_appendages: true|false
  independent_physical_parts: true|false

output_goal:
  paper_model: true|false
  easiest_manual_assembly: true|false
  preserve_shape: low|medium|high
  allow_small_parts_to_separate: true|false

paths:
  source_glb: /absolute/path/model.glb
  normalized_mesh: /absolute/path/input_UID_0.ply
  labels: /absolute/path/UID_0_NN.npy
  output_dir: /absolute/or/relative/output

partfield:
  clusters: integer
```

## Agent 必须遵守的路径规则

- 第一个模型输入必须优先使用**原始带纹理 Hunyuan3D GLB**；
- 不要把 `partfield_input_simplified.glb` 当作最终纹理来源；
- `--postprocess-only` 必须同时提供：
  - `--normalized-mesh`；
  - `--labels`；
- `labels.npy` 的聚类数必须与 `--clusters` 一致：
  - `_03.npy` → `--clusters 3`；
  - `_08.npy` → `--clusters 8`；
- 修改 `--clusters` 后必须重新运行 PartField，不能继续复用旧 labels。

---

# 3. 最重要的总决策：选择哪个 fit mode

V32 支持四种 `--fit-mode`：

```text
obb
 aabb
 shared
 primitive
```

## 3.1 快速决策表

| 用户目标 | Agent 选择 |
|---|---|
| 整个模型只要一个盒子 | `--clusters 1`；系统固定输出单个 AABB |
| Minecraft、积木、最低制作难度 | `--fit-mode aabb` |
| 方块结构，但整体是倾斜的 | `--fit-mode shared` |
| 每个部件需要独立旋转的盒子预览 | `--fit-mode obb` |
| 苹果、动物、人物等有机形体 | `--fit-mode primitive` |
| 需要标准 `paper_model.obj` | `aabb/shared/primitive + --obj-mode surface` |
| 需要所有输出用于诊断 | 支持时使用 `--obj-mode all` |

## 3.2 选择算法

```text
如果 clusters == 1：
    无论用户写什么 fit-mode，程序都会跳过 PartField并输出单个 AABB。

否则，如果目标是方块纸模或最简单装配：
    如果必须保持世界坐标轴：aabb
    如果模型整体倾斜，所有块应共享同一方向：shared
    如果只是每块独立包围盒预览：obb

否则，如果目标是保留有机轮廓：
    primitive
```

---

# 4. `--fit-mode obb`

## 4.1 定义

每个 PartField cluster 独立计算自己的 PCA/OBB 方向，每个长方体可以有不同旋转。

```text
cluster A → 独立旋转盒
cluster B → 独立旋转盒
cluster C → 独立旋转盒
```

## 4.2 适合

- 快速查看 PartField 每块的独立方向；
- 机械零件或长条部件各自方向差异很大；
- 不要求最终所有盒子构成稳定的 `surface` Paper Model。

## 4.3 不适合

- 需要严格非重叠、面接触的最终纸模；
- 需要 `paper_model.obj`；
- 苹果、人物、动物等需要曲面近似的对象。

## 4.4 OBJ 模式兼容性

| obj-mode | 是否可用 |
|---|---|
| `merged` | 是 |
| `separate` | 是 |
| `surface` | **否** |
| `all` | **否**，因为 `all` 包含 surface |

V32 会拒绝 `obb + surface/all`。

## 4.5 推荐命令

```bash
python -m partfield_mc.cli \
  "$SOURCE_GLB" \
  -o obb_result \
  --postprocess-only \
  --normalized-mesh "$NORMALIZED_MESH" \
  --labels "$LABELS" \
  --clusters "$CLUSTERS" \
  --category generic \
  --up-axis y \
  --forward-axis auto \
  --fit-mode obb \
  --obj-mode merged \
  --min-area-ratio 0 \
  --min-faces 4 \
  --grid-divisions 0 \
  --resolve-overlaps \
  --part-gap-ratio 0 \
  --face-resolution 64 \
  --surface-samples 500000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 1
```

---

# 5. `--fit-mode aabb`

## 5.1 定义

每个 cluster 拟合成与世界 X/Y/Z 轴平行的长方体。

```text
所有部件旋转 = 0
每个部件只改变中心和长宽高
```

## 5.2 适合

- Minecraft、积木、低模方块纸模；
- 优先稳定性和制作难度；
- 希望每块固定为 6 个四边面；
- 苹果或狐狸的对照基线；
- PartField 分割不够语义化但仍希望位置稳定。

## 5.3 不适合

- 高度倾斜且世界轴不合理的模型；
- 需要保留圆润外形；
- 需要苹果凹陷、动物头部等曲面细节。

## 5.4 推荐 OBJ 模式

纸模使用：

```bash
--fit-mode aabb \
--obj-mode surface \
--surface-fit-strategy refit
```

## 5.5 `surface-fit-strategy` 分类

### `refit`（推荐）

从每个 cluster 的源表面重新搜索非重叠 AABB，尽量保持源标签邻接和面接触。

```bash
--surface-fit-strategy refit
```

适合最终 Paper Model。

### `trim`（旧流程）

先拟合盒子，再按大部件优先对重叠区域做裁切。

```bash
--surface-fit-strategy trim
```

只用于复现旧结果或诊断；通常不如 `refit` 稳定。

## 5.6 `semantic-refit` 分类

| 值 | 使用场景 |
|---|---|
| `off` | 水果、产品、家具、通用物体 |
| `auto` | Agent 无法明确判断，但允许几何启发式自动判断 |
| `animal` | 狐狸、猫、狗等动物 |
| `person` | 人体、人形角色 |

它不是 LLM Agent，而是确定性几何优先级。动物/人物模式会优先保护头部或脸部，再拟合躯干。

## 5.7 自适应拆分

默认允许一个躯干 cluster 拆成两个接触长方体：

```bash
--max-extra-cuboids 1 \
--protected-min-coverage 0.85 \
--split-min-coverage-gain 0.05
```

不想增加盒子时：

```bash
--no-adaptive-split
```

## 5.8 AABB 通用 Paper Model 命令

```bash
python -m partfield_mc.cli \
  "$SOURCE_GLB" \
  -o aabb_paper_result \
  --postprocess-only \
  --normalized-mesh "$NORMALIZED_MESH" \
  --labels "$LABELS" \
  --clusters "$CLUSTERS" \
  --category generic \
  --up-axis y \
  --forward-axis auto \
  --fit-mode aabb \
  --obj-mode surface \
  --surface-fit-strategy refit \
  --refit-min-coverage 0.05 \
  --refit-beam-width 128 \
  --semantic-refit off \
  --no-adaptive-split \
  --part-gap-ratio 0 \
  --grid-divisions 0 \
  --face-resolution 64 \
  --surface-samples 500000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 1
```

## 5.9 动物 AABB 命令

```bash
python -m partfield_mc.cli \
  "$SOURCE_GLB" \
  -o animal_aabb_paper_result \
  --postprocess-only \
  --normalized-mesh "$NORMALIZED_MESH" \
  --labels "$LABELS" \
  --clusters "$CLUSTERS" \
  --category animal \
  --up-axis y \
  --forward-axis auto \
  --fit-mode aabb \
  --obj-mode surface \
  --surface-fit-strategy refit \
  --refit-min-coverage 0.05 \
  --refit-beam-width 128 \
  --semantic-refit animal \
  --max-extra-cuboids 1 \
  --protected-min-coverage 0.85 \
  --split-min-coverage-gain 0.05 \
  --part-gap-ratio 0 \
  --face-resolution 64 \
  --surface-samples 500000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 1
```

---

# 6. `--fit-mode shared`

## 6.1 定义

先计算一个模型级公共旋转，所有 cluster 都使用这个共享方向，再分别拟合长方体。

```text
模型公共方向 R
cluster A → R + 自己的尺寸和中心
cluster B → R + 自己的尺寸和中心
```

## 6.2 适合

- 整个模型相对世界轴是倾斜的；
- 建筑、家具、交通工具等大部分部件方向一致；
- 希望比 AABB 更贴合，又希望所有盒子方向统一；
- 需要 `surface` 非重叠 Paper Model。

## 6.3 不适合

- 四肢、尾巴等方向差异特别大；
- 苹果等有机主体；
- 希望每块独立 PCA 旋转时，应使用 OBB。

## 6.4 OBJ 模式兼容性

`shared` 与 `aabb` 一样支持：

```text
merged / separate / surface / all
```

## 6.5 推荐命令

```bash
python -m partfield_mc.cli \
  "$SOURCE_GLB" \
  -o shared_paper_result \
  --postprocess-only \
  --normalized-mesh "$NORMALIZED_MESH" \
  --labels "$LABELS" \
  --clusters "$CLUSTERS" \
  --category generic \
  --up-axis y \
  --forward-axis auto \
  --fit-mode shared \
  --obj-mode surface \
  --surface-fit-strategy refit \
  --refit-min-coverage 0.05 \
  --refit-beam-width 128 \
  --semantic-refit off \
  --no-adaptive-split \
  --part-gap-ratio 0 \
  --face-resolution 64 \
  --surface-samples 500000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 1
```

---

# 7. `--fit-mode primitive`

## 7.1 定义

Primitive 模式会针对每个最终物理部件，在以下程序化候选中评分：

```text
box
prism
frustum
cone
ellipsoid
convex
```

评分综合考虑几何拟合误差、目标面数差异、体积差异和纸模复杂度。

当：

```bash
--primitive-target-faces 0
```

自动目标大致为：

```text
target_faces = clamp(round(2 × sqrt(source_face_count)), 6, primitive_max_faces)
```

## 7.2 Primitive 适合

- 苹果、水果、动物、人物等有机对象；
- 希望比 AABB 更接近原始轮廓；
- 希望不同部件使用不同几何类型；
- 需要独立闭合纸模 shell；
- 需要按接触强度决定哪些部件必须连接。

## 7.3 当前限制

- 不是外部预设 OBJ/GLB 模板拟合；
- PartField 是几何分割，不保证语义分割；
- `auto` 主体可能使用 constrained surface，而不是标准 primitive；
- 结果质量仍依赖 PartField labels；
- 手工纸模应设置合理的硬面数上限。

---

# 8. Primitive 的第一层分类：`--primitive-part-mode`

支持：

```text
auto
closed
surface-patch
```

## 8.1 `auto`（通用推荐）

```bash
--primitive-part-mode auto
```

行为：

- 识别属于同一个连续主体的大面积表面分区；
- 将这些 labels 合并为一个主体；
- 主体优先使用 constrained mesh simplification；
- 细长、薄片、附肢保留为独立闭合 primitive。

适合：

- 苹果主体 + 枝叶；
- 有连续躯干和独立附肢的混合对象；
- Agent 不确定 PartField labels 是表面 patch 还是真正部件。

## 8.2 `closed`（狐狸旧行为/独立部件）

```bash
--primitive-part-mode closed
```

行为：

- 每条 PartField label 生成一个独立闭合 primitive；
- 不将多个 label 合并为 constrained 主体。

适合：

- 狐狸的头、身体、腿、尾巴已经分得较好；
- 每一块都必须成为独立纸模；
- 希望保持 V26/V27 的一 label 一 shell 逻辑。

风险：

- 如果 PartField 把同一个苹果主体切成多个表面区域，会生成多个完整实体；
- 连续主体不应盲目使用 `closed`。

## 8.3 `surface-patch`（更积极合并主体）

```bash
--primitive-part-mode surface-patch
```

行为：

- 比 `auto` 更容易把大面积相邻 labels 合并成连续主体；
- 仍尽量保护细长/薄片附肢。

适合：

- PartField 明显把一个连续主体切成多块；
- `auto` 合并不足；
- 水果、球形产品、圆润容器。

风险：

- 可能把本应独立装配的相邻大部件合并；
- 狐狸默认不要直接使用，除非确认躯干被错误切成多个表面 patch。

---

# 9. Primitive 的第二层分类：候选几何类型

通过：

```bash
--primitive-types auto
```

或：

```bash
--primitive-types ellipsoid,convex,frustum,cone,prism
```

## 9.1 类型表

| 类型 | 适合 | 不适合 |
|---|---|---|
| `box` | 方形躯干、方盒、块状零件 | 圆润水果 |
| `prism` | 长条、柱状、规则截面部件 | 强锥度部件 |
| `frustum` | 两端大小不同的躯干、肢体 | 球状主体 |
| `cone` | 尾尖、角、果梗、锥形零件 | 扁叶片 |
| `ellipsoid` | 苹果主体、头部、圆润身体 | 薄片、锐角结构 |
| `convex` | 不规则但总体凸的部件 | 明显凹陷或强非凸主体 |

## 9.2 Agent 的类型选择规则

```text
如果对象未知：primitive-types auto
如果水果主体：ellipsoid,convex,frustum,cone,prism
如果动物：auto
如果叶片/薄片很多：必须保留 convex/prism
如果模型必须方块化：不要用 primitive，改用 aabb/shared
```

---

# 10. Primitive 的第三层分类：`--primitive-contact-mode`

支持：

```text
auto
fixed
connector
move
```

## 10.1 `auto`（V32 推荐）

```bash
--primitive-contact-mode auto
```

程序对每条原始 PartField 接缝计算接触强度，然后分类：

```text
strong → fixed 共享接口
medium → connector 或 separate
weak   → 允许分离
```

适合：

- 苹果枝叶与果体接触很小；
- 狐狸耳尖、尾尖等不应被强制扩大接口；
- 同一模型同时存在强连接和弱连接；
- 希望减少因固定全部接口而造成的减面失败。

## 10.2 `fixed`

```bash
--primitive-contact-mode fixed
```

所有源相邻部件都尝试重建冻结共享接口。

适合：

- 所有连接都必须面—面装配；
- PartField 接缝可靠；
- 独立部件之间接口较大。

风险：

- 点接触、短边接触也会被强制重建；
- 苹果枝叶等弱连接可能拖累主体减面；
- 复杂接口更容易触发修复或回退。

## 10.3 `connector`

```bash
--primitive-contact-mode connector
```

保持主部件的位置和方向，不移动主体；在相邻部件之间生成闭合连接件。

适合：

- 希望所有部件形成装配体；
- 不希望为了接触而旋转/移动主部件；
- 允许额外纸质连接件。

## 10.4 `move`（旧模式）

```bash
--primitive-contact-mode move
```

通过移动或旋转子部件恢复接触。

仅用于旧结果兼容或实验。默认 Agent 不应选择，因为可能改变部件位置和姿态。

---

# 11. Auto Contact 的接触强度算法

## 11.1 指标

对每条接缝计算：

```text
area_ratio  = 接口面积 / 较小部件表面积
seam_ratio  = 接缝长度 / sqrt(较小部件表面积)
edge_ratio  = 接触边数量 / sqrt(较小部件面数)
point_ratio = 唯一接触点数量 / sqrt(较小部件面数)
```

归一化得分：

```text
score =
  0.50 × clamp(area_ratio / 0.10, 0, 1)
+ 0.25 × clamp(seam_ratio / 0.85, 0, 1)
+ 0.15 × clamp(edge_ratio / 2.0, 0, 1)
+ 0.10 × clamp(point_ratio / 2.0, 0, 1)
```

若接触边数或唯一接触点数小于：

```bash
--primitive-contact-min-edge-count
```

则强制分类为 `weak`。

## 11.2 分类规则

```text
forced_weak 或 score < weak_threshold → weak
score >= strong_threshold            → strong
其他                                  → medium
```

必须满足：

```text
0 <= weak_threshold < strong_threshold <= 1
```

## 11.3 Medium 处理

```bash
--primitive-contact-medium-mode connector
```

中等连接添加连接件。

```bash
--primitive-contact-medium-mode separate
```

中等连接也允许分离。

## 11.4 推荐阈值预设

### 苹果、枝叶、细梗：允许弱/中等连接分离

```bash
--primitive-contact-mode auto \
--primitive-contact-weak-threshold 0.28 \
--primitive-contact-strong-threshold 0.62 \
--primitive-contact-min-edge-count 8 \
--primitive-contact-medium-mode separate
```

### 狐狸、动物：中等连接使用 connector

```bash
--primitive-contact-mode auto \
--primitive-contact-weak-threshold 0.20 \
--primitive-contact-strong-threshold 0.55 \
--primitive-contact-min-edge-count 6 \
--primitive-contact-medium-mode connector
```

### 强制保留更多连接

```bash
--primitive-contact-mode auto \
--primitive-contact-weak-threshold 0.12 \
--primitive-contact-strong-threshold 0.45 \
--primitive-contact-min-edge-count 4 \
--primitive-contact-medium-mode connector
```

### 只保留非常可靠的强连接

```bash
--primitive-contact-mode auto \
--primitive-contact-weak-threshold 0.35 \
--primitive-contact-strong-threshold 0.75 \
--primitive-contact-min-edge-count 10 \
--primitive-contact-medium-mode separate
```

---

# 12. Constrained Surface 减面参数

这些参数只在 `primitive-part-mode auto/surface-patch` 识别出连续主体时有意义。

## 12.1 关键参数

| 参数 | 含义 | Agent 推荐 |
|---|---|---|
| `--primitive-surface-main-body-min-area-ratio` | 最大非薄片组占总表面积至少多少才视为主体 | `0.35` |
| `--primitive-surface-boundary-rings` | 接口周围额外冻结的顶点环数 | `0`；接口需更稳时 `1` |
| `--primitive-surface-search-steps` | 减面搜索分辨率数量 | `18–24` |
| `--primitive-surface-min-reduction-ratio` | 希望至少减少的面数比例 | `0.50` |
| `--primitive-surface-hard-max-faces` | 单个 constrained 主体硬面数上限 | 手工纸模 `128–256`；预览 `512` |

## 12.2 面数建议

| 目标 | hard max faces |
|---|---:|
| 非常简单、容易手工制作 | `64–96` |
| 普通纸模 | `128` |
| 保留更多水果/动物轮廓 | `256` |
| 仅用于数字预览 | `512` |

Agent 不应默认把几万面结果送入纹理阶段。

## 12.3 Primitive 面数参数区别

```text
--primitive-max-faces
```

控制标准 `box/prism/frustum/cone/ellipsoid/convex` 单部件面数。

```text
--primitive-surface-hard-max-faces
```

控制 constrained surface 主体面数。

两者不是同一个参数。

---

# 13. Primitive 接口与验证参数

## 13.1 固定接口

```bash
--primitive-interface-max-sides 8
```

冻结接口多边形最大边数。推荐：

- 简单纸模：`6–8`；
- 苹果主体：`8–12`；
- 复杂接口：最大可提高，但制作难度也会提高。

```bash
--primitive-interface-min-width-ratio 0.006
```

源接缝退化为线或点时，备用接口最小尺寸。细梗可用 `0.004`。

```bash
--primitive-interface-plane-tolerance-ratio 0.000001
```

只是数值验证容差。不要用持续放大容差来掩盖接口拓扑错误。

## 13.2 Validation policy

### `repair`（默认推荐）

```bash
--primitive-validation-policy repair
```

尝试修复接口；几何完全一致但侧向分类有歧义时记录警告并继续。

### `warn`

```bash
--primitive-validation-policy warn
```

尽量不终止，只记录问题。适合批量生成，但必须检查 JSON。

### `strict`

```bash
--primitive-validation-policy strict
```

任何固定接口验证失败都终止。只用于测试和算法调试。

Agent 面向普通生成任务时默认使用 `repair`。

---

# 14. Primitive Connector 参数

仅当接触策略需要 connector 时使用：

```bash
--primitive-connector-sides 4 \
--primitive-connector-radius-ratio 0.028 \
--primitive-connector-inset-ratio 0.28 \
--primitive-connector-min-length-ratio 0.002
```

| 参数 | 含义 |
|---|---|
| `connector-sides` | 连接件横截面边数；4 最适合纸模 |
| `radius-ratio` | 相对模型最长边的目标半径 |
| `inset-ratio` | 接触中心向面内部移动，避免边/角接触 |
| `min-length-ratio` | 两个面很近时仍保留的最小连接长度 |

苹果允许枝叶分离时，使用：

```bash
--primitive-contact-medium-mode separate
```

这样通常不需要 connector。

---

# 15. `--obj-mode` 完整分类

## 15.1 `merged`

```bash
--obj-mode merged
```

- 所有部件写入一个 OBJ；
- 不一定生成 canonical `paper_model.obj`；
- 适合普通预览。

## 15.2 `separate`

```bash
--obj-mode separate
```

- 每个部件输出独立 OBJ；
- 适合分别编辑和检查。

## 15.3 `surface`

```bash
--obj-mode surface
```

- 推荐 Paper Model 模式；
- AABB/shared：进行 constrained non-overlap refit；
- Primitive：输出闭合 primitive shells 和 `paper_model.obj`；
- OBB 不支持。

## 15.4 `all`

```bash
--obj-mode all
```

- 输出 merged、surface、separate 和 paper model 结果；
- 用于调试和对比；
- 文件更多、处理更慢；
- OBB 不支持。

## 15.5 兼容矩阵

| fit-mode | merged | separate | surface | all |
|---|---:|---:|---:|---:|
| `obb` | ✓ | ✓ | ✗ | ✗ |
| `aabb` | ✓ | ✓ | ✓ | ✓ |
| `shared` | ✓ | ✓ | ✓ | ✓ |
| `primitive` | ✓ | ✓ | ✓ | ✓ |

---

# 16. Agent 不能混用的参数

## 16.1 AABB/shared/OBB 不要传 Primitive 参数

以下参数只属于 Primitive：

```text
--primitive-*
--no-primitive-*
```

## 16.2 Primitive 不要依赖 Cuboid surface 参数

以下参数主要属于 AABB/shared `surface` 流程：

```text
--surface-fit-strategy
--refit-min-coverage
--refit-beam-width
--no-refit-preserve-contact
--semantic-refit
--no-adaptive-split
--max-extra-cuboids
--protected-min-coverage
--split-min-coverage-gain
```

Primitive 命令中不要把这些参数当作有效调节手段。

## 16.3 `clusters=1` 的特殊规则

```bash
--clusters 1
```

会跳过 PartField并固定生成单个 AABB。即使同时写：

```bash
--fit-mode primitive
```

也不会生成 Primitive。

## 16.4 Postprocess-only 规则

只运行 FitOBJ 时必须写：

```bash
--postprocess-only \
--normalized-mesh "$NORMALIZED_MESH" \
--labels "$LABELS"
```

此时不要再加入：

```text
--image-front
--image-right
--multiview
--texture
--hunyuan-repo
```

纹理不会被取消；程序会读取已有的带纹理 `SOURCE_GLB` 并重新烘焙纸模纹理。

---

# 17. Agent 自动选参决策树

下面是 Agent 应执行的确定性流程。

```text
STEP 1：判断运行阶段

如果只有图片：
    运行 Hunyuan3D + PartField + FitOBJ
    必须保留 --texture

如果已有带纹理 GLB，但没有 PLY/NPY：
    运行 PartField + FitOBJ

如果已有 GLB + normalized mesh + labels：
    使用 --postprocess-only，只运行 FitOBJ


STEP 2：判断 clusters

如果用户明确指定：使用用户值
否则：
    整体单盒：1
    水果主体+梗+叶：3，必要时尝试 4
    简单动物：6–10，狐狸默认 8
    人物：8–14
    产品：根据可见结构数量


STEP 3：判断 fit-mode

如果只需要一个盒子：clusters=1
否则如果目标是方块纸模：aabb
否则如果所有方块应共享一个倾斜方向：shared
否则如果仅查看每块独立方向：obb
否则：primitive


STEP 4：判断 obj-mode

如果最终用于 Blender Paper Model：surface
如果需要每块单独文件：separate
如果只预览：merged
如果调试全部阶段：all


STEP 5：Primitive part mode

如果连续主体 + 少量附肢：auto
如果每个 label 已经是真正独立部件：closed
如果主体明显被 PartField 切成多个大表面区域：surface-patch


STEP 6：Primitive contact mode

一般情况：auto
所有接口都必须固定：fixed
所有相邻块用额外连接件：connector
仅复现旧移动算法：move


STEP 7：Auto contact 预设

水果枝叶可分离：
    weak=0.28, strong=0.62, min_edges=8, medium=separate

动物装配：
    weak=0.20, strong=0.55, min_edges=6, medium=connector


STEP 8：面数

标准 primitive：
    简单 32–48
    普通 48–72
    复杂 72–96

constrained 主体 hard max：
    手工纸模 128
    形状优先 256
    数字预览 512


STEP 9：纹理

总面数预算较低：face-resolution 96
普通：64
面数较多：48
调试：32
surface-samples 默认 500000
```

---

# 18. Agent 输出格式规范

Agent 应先给出选参判断，再给出一条可以复制的完整命令。

建议输出 JSON：

```json
{
  "run_stage": "fitobj_only",
  "fit_mode": "primitive",
  "obj_mode": "surface",
  "reason": "continuous organic body with thin appendages",
  "parameters": {
    "clusters": 3,
    "category": "generic",
    "primitive_part_mode": "auto",
    "primitive_contact_mode": "auto",
    "primitive_contact_weak_threshold": 0.28,
    "primitive_contact_strong_threshold": 0.62,
    "primitive_contact_min_edge_count": 8,
    "primitive_contact_medium_mode": "separate",
    "primitive_max_faces": 72,
    "primitive_surface_hard_max_faces": 256,
    "primitive_validation_policy": "repair"
  },
  "requires_partfield_rerun": false,
  "command": "python -m partfield_mc.cli ..."
}
```

Agent 不得：

- 输出 CLI 中不存在的枚举；
- 猜测不存在的文件路径；
- 把 `_03.npy` 配成 `--clusters 8`；
- 在 `obb` 下选择 `surface/all`；
- 在 `clusters=1` 时声称运行 Primitive；
- 声称 V32 已加载外部 primitive 模板库；
- 因为 `--postprocess-only` 没有 `--texture` 就声称纹理被禁用。

---

# 19. 完整预设一：苹果/水果主体 + 枝叶

目标：

- 苹果主体保持连续；
- 主体使用 constrained surface；
- 枝叶接触弱时允许分离；
- 不让弱接口把主体拖到数万面；
- 输出 Paper Model。

```bash
rm -rf apple_primitive_v32

python -m partfield_mc.cli \
  "$SOURCE_GLB" \
  -o apple_primitive_v32 \
  --postprocess-only \
  --normalized-mesh "$NORMALIZED_MESH" \
  --labels "$LABELS" \
  --clusters 3 \
  --category generic \
  --up-axis y \
  --forward-axis auto \
  --fit-mode primitive \
  --obj-mode surface \
  --primitive-part-mode auto \
  --primitive-types ellipsoid,convex,frustum,cone,prism \
  --primitive-target-faces 0 \
  --primitive-max-faces 72 \
  --primitive-max-sides 24 \
  --primitive-fit-samples 3500 \
  --primitive-complexity-weight 0.012 \
  --primitive-contact-mode auto \
  --primitive-contact-weak-threshold 0.28 \
  --primitive-contact-strong-threshold 0.62 \
  --primitive-contact-min-edge-count 8 \
  --primitive-contact-medium-mode separate \
  --primitive-contact-overlap-ratio 0 \
  --primitive-interface-max-sides 12 \
  --primitive-interface-min-width-ratio 0.004 \
  --primitive-interface-plane-tolerance-ratio 0.000001 \
  --primitive-surface-main-body-min-area-ratio 0.35 \
  --primitive-surface-boundary-rings 0 \
  --primitive-surface-search-steps 24 \
  --primitive-surface-min-reduction-ratio 0.50 \
  --primitive-surface-hard-max-faces 256 \
  --primitive-validation-policy repair \
  --face-resolution 64 \
  --surface-samples 500000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 2
```

若 `auto` 没有合并完整果体，先尝试：

```bash
--primitive-part-mode surface-patch
```

不要第一时间放大接口容差。

---

# 20. 完整预设二：狐狸/动物独立闭合部件

目标：

- 头、身体、腿、尾巴保持独立闭合 shell；
- 强连接固定；
- 中等连接使用 connector；
- 很弱的点接触允许分离。

```bash
rm -rf fox_primitive_v32

python -m partfield_mc.cli \
  "$SOURCE_GLB" \
  -o fox_primitive_v32 \
  --postprocess-only \
  --normalized-mesh "$NORMALIZED_MESH" \
  --labels "$LABELS" \
  --clusters 8 \
  --category animal \
  --up-axis y \
  --forward-axis auto \
  --fit-mode primitive \
  --obj-mode surface \
  --primitive-part-mode closed \
  --primitive-types auto \
  --primitive-target-faces 0 \
  --primitive-max-faces 48 \
  --primitive-max-sides 24 \
  --primitive-fit-samples 2500 \
  --primitive-complexity-weight 0.025 \
  --primitive-contact-mode auto \
  --primitive-contact-weak-threshold 0.20 \
  --primitive-contact-strong-threshold 0.55 \
  --primitive-contact-min-edge-count 6 \
  --primitive-contact-medium-mode connector \
  --primitive-contact-overlap-ratio 0 \
  --primitive-interface-max-sides 8 \
  --primitive-interface-min-width-ratio 0.006 \
  --primitive-interface-plane-tolerance-ratio 0.000001 \
  --primitive-connector-sides 4 \
  --primitive-connector-radius-ratio 0.028 \
  --primitive-connector-inset-ratio 0.28 \
  --primitive-connector-min-length-ratio 0.002 \
  --primitive-validation-policy repair \
  --face-resolution 64 \
  --surface-samples 500000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 1
```

若 PartField 明显把狐狸躯干切成两个大表面区域，可试：

```bash
--primitive-part-mode auto
```

但应先检查 segmented PLY。

---

# 21. 完整预设三：通用有机模型

```bash
python -m partfield_mc.cli \
  "$SOURCE_GLB" \
  -o generic_primitive_v32 \
  --postprocess-only \
  --normalized-mesh "$NORMALIZED_MESH" \
  --labels "$LABELS" \
  --clusters "$CLUSTERS" \
  --category generic \
  --up-axis y \
  --forward-axis auto \
  --fit-mode primitive \
  --obj-mode surface \
  --primitive-part-mode auto \
  --primitive-types auto \
  --primitive-target-faces 0 \
  --primitive-max-faces 64 \
  --primitive-max-sides 24 \
  --primitive-fit-samples 2500 \
  --primitive-complexity-weight 0.020 \
  --primitive-contact-mode auto \
  --primitive-contact-weak-threshold 0.22 \
  --primitive-contact-strong-threshold 0.58 \
  --primitive-contact-min-edge-count 6 \
  --primitive-contact-medium-mode connector \
  --primitive-interface-max-sides 8 \
  --primitive-interface-min-width-ratio 0.006 \
  --primitive-interface-plane-tolerance-ratio 0.000001 \
  --primitive-surface-main-body-min-area-ratio 0.35 \
  --primitive-surface-boundary-rings 0 \
  --primitive-surface-search-steps 20 \
  --primitive-surface-min-reduction-ratio 0.50 \
  --primitive-surface-hard-max-faces 256 \
  --primitive-validation-policy repair \
  --face-resolution 64 \
  --surface-samples 500000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 1
```

---

# 22. 完整预设四：单个全局 AABB

`clusters=1` 会直接跳过 PartField：

```bash
python -m partfield_mc.cli \
  "$SOURCE_GLB" \
  -o single_aabb_result \
  --clusters 1 \
  --up-axis y \
  --fit-mode aabb \
  --obj-mode merged \
  --face-resolution 96 \
  --surface-samples 500000 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 1
```

不需要 `--normalized-mesh` 或 `--labels`。

---

# 23. Stage 1：单独运行 Hunyuan3D

## 23.1 环境变量

```bash
conda activate partfield

export PROJECT=/mnt/e/yp/partfield_hunyuan_v32
export HUNYUAN_REPO=/mnt/e/yp/Hunyuan3D-2
export PARTFIELD_REPO=/mnt/e/yp/PartField
export CHECKPOINT=/mnt/e/yp/PartField/model/model_objaverse.ckpt
export WORK=$PROJECT/model_stages

mkdir -p "$WORK/01_hunyuan" "$WORK/02_partfield" "$WORK/03_fitobj"
```

## 23.2 多视角带纹理

```bash
export FRONT_IMAGE=/absolute/path/front.png
export RIGHT_IMAGE=/absolute/path/right.png
export SOURCE_GLB=$WORK/01_hunyuan/hunyuan3d_textured_multiview.glb

python - <<'PY'
import os
from partfield_mc.hunyuan3d import generate_mesh

generate_mesh(
    image=None,
    images={
        "front": os.environ["FRONT_IMAGE"],
        "right": os.environ["RIGHT_IMAGE"],
    },
    output=os.environ["SOURCE_GLB"],
    hunyuan_repo=os.environ["HUNYUAN_REPO"],
    texture=True,
    multiview=True,
)
PY
```

`texture=True` 会加载 Hunyuan Paint 模型。若 Shape + Paint 同一进程内存不足，应分开进程处理，但最终 FitOBJ 的输入仍应使用带纹理 GLB。

---

# 24. Stage 2：单独运行 PartField

```bash
export CLUSTERS=3
export SIMPLIFY_FACES=8000

python - <<'PY'
import os
import shlex
from pathlib import Path

from partfield_mc.mesh_io import simplify_mesh_for_partfield
from partfield_mc.partfield_runner import PartFieldRunConfig, run_partfield

source_glb = Path(os.environ["SOURCE_GLB"]).expanduser().resolve()
output_dir = Path(os.environ["WORK"]).expanduser().resolve() / "02_partfield"
output_dir.mkdir(parents=True, exist_ok=True)

simplified_glb = simplify_mesh_for_partfield(
    input_path=source_glb,
    output_path=output_dir / "partfield_input_simplified.glb",
    target_faces=int(os.environ.get("SIMPLIFY_FACES", "5000")),
    up_axis="y",
)

artifacts = run_partfield(
    input_path=simplified_glb,
    output_dir=output_dir,
    config=PartFieldRunConfig(
        repo=Path(os.environ["PARTFIELD_REPO"]),
        checkpoint=Path(os.environ["CHECKPOINT"]),
        clusters=int(os.environ["CLUSTERS"]),
        clustering="agglo",
        adjacency="mst",
        n_point_per_face=50,
        n_sample_each=1000,
        preprocess_mesh=False,
        up_axis="y",
        force=False,
    ),
)

env_file = output_dir / "partfield_artifacts.env"
lines = [
    f"export SOURCE_GLB={shlex.quote(str(source_glb))}",
    f"export PARTFIELD_INPUT_SIMPLIFIED={shlex.quote(str(simplified_glb))}",
    f"export NORMALIZED_MESH={shlex.quote(str(artifacts.normalized_mesh))}",
    f"export LABELS={shlex.quote(str(artifacts.labels_path))}",
]
if artifacts.colored_segmentation is not None:
    lines.append(f"export SEGMENTED_PLY={shlex.quote(str(artifacts.colored_segmentation))}")
env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"NORMALIZED_MESH={artifacts.normalized_mesh}")
print(f"LABELS={artifacts.labels_path}")
print(f"ARTIFACT_ENV={env_file}")
PY
```

推荐 `simplify_faces`：

| 模型 | 推荐 |
|---|---:|
| 简单方块 | `3000–5000` |
| 一般动物 | `5000` |
| 苹果、细梗、薄叶 | `8000` |
| 更细结构 | `10000`，但 PartField 开销更高 |

---

# 25. Stage 3：只运行 FitOBJ

```bash
source "$WORK/02_partfield/partfield_artifacts.env"

ls -lh "$SOURCE_GLB" "$NORMALIZED_MESH" "$LABELS"
```

然后选择第 5、6、19、20 或 21 节中的命令，并保留：

```bash
--postprocess-only \
--normalized-mesh "$NORMALIZED_MESH" \
--labels "$LABELS"
```

---

# 26. 一条命令运行全部阶段

图片直接生成带纹理模型、PartField、Primitive：

```bash
python -m partfield_mc.cli \
  --image-front /absolute/path/front.png \
  --image-right /absolute/path/right.png \
  --multiview \
  --texture \
  --hunyuan-repo /mnt/e/yp/Hunyuan3D-2 \
  --partfield-repo /mnt/e/yp/PartField \
  --checkpoint /mnt/e/yp/PartField/model/model_objaverse.ckpt \
  --simplify-faces 8000 \
  --n-point-per-face 50 \
  --n-sample-each 1000 \
  -o full_primitive_result \
  --clusters 3 \
  --clustering agglo \
  --adjacency mst \
  --category generic \
  --up-axis y \
  --forward-axis auto \
  --fit-mode primitive \
  --obj-mode surface \
  --primitive-part-mode auto \
  --primitive-types auto \
  --primitive-max-faces 72 \
  --primitive-max-sides 24 \
  --primitive-contact-mode auto \
  --primitive-contact-weak-threshold 0.28 \
  --primitive-contact-strong-threshold 0.62 \
  --primitive-contact-min-edge-count 8 \
  --primitive-contact-medium-mode separate \
  --primitive-surface-hard-max-faces 256 \
  --primitive-validation-policy repair \
  --face-resolution 64 \
  --surface-samples 500000 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 2
```

如果 Hunyuan Paint 加载时出现 `Cannot allocate memory`，不是 FitOBJ 参数错误，而是 Shape + Paint 的系统内存峰值问题。

---

# 27. 纹理参数自动选择

## 27.1 `face-resolution`

| 预计总纸模面数 | 推荐值 |
|---|---:|
| ≤ 100 | `96` |
| 101–300 | `64` |
| 301–800 | `48` |
| > 800 | `32`，并警告模型可能过于复杂 |

## 27.2 其他参数

```bash
--surface-samples 500000
--palette-size 0
--texture-filter bilinear
--uv-wrap clamp
--padding 1
```

图片边缘需要更强隔离时用：

```bash
--padding 2
```

`palette-size 0` 表示不做调色板量化。

---

# 28. 输出文件

## Paper Model 推荐文件

```text
paper_model.obj
paper_model.mtl
paper_model_texture.png
paper_model.glb
paper_model_parts.json
```

- Blender Paper Model 展开：`paper_model.obj`；
- 纹理预览：`paper_model.glb`；
- Agent 自动检查：`paper_model_parts.json`。

## Primitive JSON 重点字段

```text
primitive_type
paper_face_count
triangle_count
fitting_strategy
source_segment_ids
contact_strength_classification
allowed_separated_contact_edges
required_contact_graph_connected
spatially_connected_assembly
connector_count
```

在 `auto` 接触模式下：

```text
required_contact_graph_connected = true
```

表示所有强制连接都满足。

```text
spatially_connected_assembly = false
```

不一定是错误；可能只是 weak/medium 接触被允许分离。

---

# 29. Agent 运行后自动检查规则

Agent 应读取日志和 `paper_model_parts.json`。

## 29.1 必须检查

```text
paper_face_count
primitive_type
fitting_strategy
allowed_separated_contact_edges
contact_strength_classification
mandatory_reduction_satisfied / budget_ok
```

## 29.2 警告条件

- 单个 constrained surface 超过设定 hard max；
- 总纸模面数超过 `800`；
- `budget_ok=false`；
- `mandatory_reduction_failed=true`；
- `contact_required_graph_connected=false`；
- 大量部件全部选择 ellipsoid；
- PartField 大 cluster 被拟合成多个完整实体；
- texture atlas 面数达到数千以上。

## 29.3 自动补救顺序

```text
1. 检查 segmented PLY 是否分割错误；
2. 不改 clusters 时，优先只重跑 FitOBJ；
3. 连续主体：closed → auto → surface-patch；
4. 弱附肢：medium connector → medium separate；
5. 主体面数高：降低 hard-max-faces；
6. 方块基线：切换 aabb；
7. 只有 labels 本身不合理时才重跑 PartField并修改 clusters。
```

---

# 30. 常见错误

## `--obj-mode surface requires --fit-mode aabb or shared`

说明使用了：

```text
obb + surface/all
```

修复：改用 `aabb/shared`，或把 obj-mode 改为 `merged/separate`。

## `--postprocess-only requires --normalized-mesh and --labels`

补充两个路径参数。

## labels 与 clusters 不匹配

重新使用对应 `_NN.npy`，或重跑 PartField clustering。

## PrimitiveTexture 面数非常大

例如：

```text
[PrimitiveTexture] 1/31971
```

应停止，检查：

```text
primitive-surface-hard-max-faces
budget_ok
mandatory_reduction_failed
```

## Hunyuan Paint `Cannot allocate memory`

增加 WSL 内存/swap，或分开 Shape 和 Paint 进程；不要通过去掉 `--texture` 来改变用户需求。

---

# 31. 最终推荐默认值

## Agent 默认 Cuboid Paper Model

```bash
--fit-mode aabb \
--obj-mode surface \
--surface-fit-strategy refit \
--refit-min-coverage 0.05 \
--refit-beam-width 128 \
--semantic-refit off \
--part-gap-ratio 0
```

## Agent 默认 Organic Paper Model

```bash
--fit-mode primitive \
--obj-mode surface \
--primitive-part-mode auto \
--primitive-types auto \
--primitive-target-faces 0 \
--primitive-max-faces 64 \
--primitive-max-sides 24 \
--primitive-contact-mode auto \
--primitive-contact-weak-threshold 0.22 \
--primitive-contact-strong-threshold 0.58 \
--primitive-contact-min-edge-count 6 \
--primitive-contact-medium-mode connector \
--primitive-surface-hard-max-faces 256 \
--primitive-validation-policy repair
```

## Agent 默认水果/枝叶

```bash
--fit-mode primitive \
--obj-mode surface \
--primitive-part-mode auto \
--primitive-contact-mode auto \
--primitive-contact-weak-threshold 0.28 \
--primitive-contact-strong-threshold 0.62 \
--primitive-contact-min-edge-count 8 \
--primitive-contact-medium-mode separate \
--primitive-surface-hard-max-faces 256 \
--primitive-validation-policy repair
```

## Agent 默认狐狸/动物独立部件

```bash
--fit-mode primitive \
--obj-mode surface \
--primitive-part-mode closed \
--primitive-contact-mode auto \
--primitive-contact-weak-threshold 0.20 \
--primitive-contact-strong-threshold 0.55 \
--primitive-contact-min-edge-count 6 \
--primitive-contact-medium-mode connector \
--primitive-validation-policy repair
```

---

# 32. 一句话总结

```text
方块纸模选 AABB；整体倾斜方块选 Shared；独立方向盒子预览选 OBB；
有机形体选 Primitive。Primitive 中连续主体选 auto/surface-patch，真正独立部件选 closed；
接触默认选 auto，由 strong/medium/weak 决定固定接口、connector 或允许分离。
```
