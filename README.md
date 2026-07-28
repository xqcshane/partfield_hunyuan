# Hunyuan3D → PartField → FitOBJ：V33 完整运行与 Agent 自动选参说明

> 适用版本：`partfield-mc-pipeline 0.33.0`
>
> 主入口：
>
> ```bash
> python -m partfield_mc.cli
> ```

本 README 面向两类使用者：

1. 人工在 Ubuntu / WSL2 中运行 Hunyuan3D、PartField 和 FitOBJ；
2. 将本文交给 AI Agent，由 Agent 根据对象类型、输入文件、目标风格和装配要求自动生成完整参数。

本文以 **V33 实际 CLI 和代码行为**为准。V33 相比 V32 的核心变化是：

- Primitive 默认使用拓扑锁定的规整模板：`--primitive-template-mode regular`；
- 规整模板不再被 PartField 接口切割、局部拉扯或 constrained-surface 回退破坏；
- 接触只通过模板已有 attachment face 或受尺寸限制的小 connector 恢复；
- 增加空间近接触推断，能够识别“视觉接触但拓扑没有共享边”的瓶盖、肢体、耳朵和尾巴；
- 保留旧的自适应局部变形算法，但必须显式使用 `--primitive-template-mode adaptive`。

---

# 0. 必须先理解的事实

## 0.1 当前 Primitive 不是外部 OBJ/GLB 预设模型库

V33 的 Primitive 是程序在运行时生成的 canonical template：

```text
box
prism
frustum
cone
ellipsoid
convex
```

当前流程不是：

```text
读取 primitive_presets/*.obj
→ 选择一个预设网格
→ 配准和形变
```

而是：

```text
根据 PartField 部件点云生成多个参数化候选
→ 旋转、平移、非均匀缩放到目标部件
→ 计算拟合分数
→ 选择最佳候选
```

因此 Agent 不得声称“系统已加载外部预设 3D 模型”。

## 0.2 V33 的 `regular` 与旧版 `adaptive` 完全不同

### `regular`：V33 默认，优先稳定、规整、低面纸模

```bash
--primitive-template-mode regular
```

它保证：

- `box / prism / frustum / cone / ellipsoid` 的 canonical topology 不被改变；
- 不把任意 PartField 接口多边形切入主体；
- 不使用 constrained-surface vertex clustering；
- 不创建 zipper strip 或大面积局部补片；
- prism、frustum、cone 保持同轴；
- 接触使用已有 attachment face 或小型 connector；
- connector 面积受到硬限制，避免出现大白色法兰；
- 接触处理后检查模板边长签名，确认模板没有被局部变形。

### `adaptive`：旧版兼容，优先原始表面和冻结接口

```bash
--primitive-template-mode adaptive
```

它允许：

- surface-patch 合并；
- constrained mesh simplification；
- frozen interface polygon；
- 局部接口适配和非凸补片。

但它可能出现：

- 主体塌陷；
- 斜切大平面；
- 宽大白色接口；
- 减面失败后面数过高；
- 细部正反面被错误合并。

**Agent 默认必须选 `regular`。只有用户明确要求保持原始复杂曲面或旧版冻结接口时，才选择 `adaptive`。**

## 0.3 PartField 是表面分割，不是可靠语义分割

PartField 输出的是 `segment_id`，不保证：

```text
segment 1 = 头
segment 2 = 身体
segment 3 = 腿
```

它可能把同一个苹果主体分成多个表面区域，也可能把叶片和局部果皮放入同一个 label。

因此 Agent 必须先检查：

```text
partfield_clusters/.../ply/*_0_NN.ply
```

这个文件只用于查看分割结果，**不能作为 `--normalized-mesh`**。

---

# 1. 完整流水线

```text
图片
  ↓ Stage 1：Hunyuan3D Shape
高精度几何
  ↓ Hunyuan Paint（使用 --texture）
带纹理 Hunyuan GLB
  ↓ Stage 2：PartField
normalized_mesh.ply + labels.npy
  ↓ Stage 3：FitOBJ
OBB / AABB / Shared Cuboid / Regular Primitive Paper Model
```

三种运行范围：

| 当前已有文件 | 运行范围 | CLI 方式 |
|---|---|---|
| 只有图片 | Hunyuan3D + PartField + FitOBJ | 使用 `--image-* --multiview --texture` |
| 已有带纹理 GLB | PartField + FitOBJ | 把 GLB 作为第一个位置参数 |
| 已有带纹理 GLB、normalized PLY、labels NPY | 只运行 FitOBJ | 使用 `--postprocess-only` |

修改 FitOBJ 参数或代码后，应优先只重跑 Stage 3，不要反复运行 Hunyuan3D 和 PartField。

---

# 2. 安装与目录

推荐：

- Ubuntu 或 WSL2；
- Python 3.10；
- NVIDIA GPU；
- 已安装 Hunyuan3D-2；
- 已安装 NVIDIA PartField 和 checkpoint。

示例：

```bash
conda activate partfield

cd /mnt/e/yp/partfield_hunyuan_v33
python -m pip install -r requirements_wrapper.txt
python -m pip install -e . --no-build-isolation
```

检查 CLI：

```bash
python -m partfield_mc.cli --help
```

常用目录变量：

```bash
export PROJECT=/mnt/e/yp/partfield_hunyuan_v33
export HUNYUAN_REPO=/mnt/e/yp/Hunyuan3D-2
export PARTFIELD_REPO=/mnt/e/yp/PartField
export CHECKPOINT=/mnt/e/yp/PartField/model/model_objaverse.ckpt
```

---

# 3. Windows 与 WSL 路径转换

Windows：

```text
E:\yp\test\puppy\front.png
```

WSL：

```text
/mnt/e/yp/test/puppy/front.png
```

Agent 输出给 WSL 的命令时，必须把：

```text
E:\...
```

转换为：

```text
/mnt/e/...
```

包含空格或括号的路径必须加引号：

```bash
--image-front "/mnt/e/yp/test/puppy/front(1).png"
```

---

# 4. Stage 3 的三个关键文件

只运行 FitOBJ 时，必须有：

```text
SOURCE_GLB
NORMALIZED_MESH
LABELS
```

## 4.1 `SOURCE_GLB`

应使用原始带纹理 Hunyuan3D 模型：

```text
hunyuan3d_textured_multiview.glb
```

不要用以下文件作为最终纹理来源：

```text
partfield_input_simplified.glb
PartField 工作目录中的无纹理 GLB
已拟合后的 paper_model.glb
```

正确示例：

```bash
SOURCE_GLB=/mnt/e/yp/partfield_hunyuan_v32/milk_aabb_result/hunyuan3d_textured_multiview.glb
```

## 4.2 `LABELS`

PartField 聚类结果：

```text
partfield_clusters/UID/cluster_out/UID_0_NN.npy
```

示例：

```bash
LABELS=/mnt/e/yp/partfield_hunyuan_v32/milk_aabb_result/partfield_clusters/partfield_input_simplified_5261707094/cluster_out/partfield_input_simplified_5261707094_0_03.npy
```

文件名中的聚类数必须匹配：

```text
_0_03.npy → --clusters 3
_0_08.npy → --clusters 8
```

修改 `--clusters` 后必须重新运行 PartField clustering，不能复用旧 NPY。

## 4.3 `NORMALIZED_MESH`

这是最容易填错的参数。

正确文件是 PartField `exp_results` 中，与 labels 面顺序严格对应的：

```text
input_UID_0.ply
```

示例：

```bash
NORMALIZED_MESH=/mnt/e/yp/PartField/exp_results/partfield_features/mc_pipeline/partfield_input_simplified_5261707094/input_partfield_input_simplified_5261707094_0.ply
```

### 以下文件都不是 `NORMALIZED_MESH`

PartField 工作输入 GLB：

```text
.../partfield_work/UID/data/UID.glb
```

彩色分割预览 PLY：

```text
.../partfield_clusters/UID/ply/UID_0_NN.ply
```

减面输入：

```text
.../partfield_input_simplified.glb
```

它们可能存在不同的：

- 顶点顺序；
- 面顺序；
- 面数量；
- 坐标归一化。

强行混用会让 label 被映射到错误三角面，造成 primitive 错位、塌陷和纹理错误。

### 自动查找 `NORMALIZED_MESH`

```bash
CASE=partfield_input_simplified_5261707094

NORMALIZED_MESH=$(find /mnt/e/yp/PartField/exp_results \
  -type f \
  -name "input_${CASE}_0.ply" \
  | head -n 1)

echo "$NORMALIZED_MESH"
```

或全盘查找：

```bash
find /mnt/e/yp \
  -type f \
  -name "input_partfield_input_simplified_5261707094_0.ply" \
  2>/dev/null
```

### 运行前强制检查

```bash
for FILE in "$SOURCE_GLB" "$NORMALIZED_MESH" "$LABELS"; do
  if [ ! -f "$FILE" ]; then
    echo "ERROR: file not found: [$FILE]"
    exit 1
  fi
done

ls -lh "$SOURCE_GLB" "$NORMALIZED_MESH" "$LABELS"
```

---

# 5. Hunyuan3D 图片参数

支持：

```text
--image                 单图
--image-front           正面
--image-left            左侧
--image-right           右侧
--image-side            左侧别名
--image-back            背面
```

当前没有：

```text
--image-top
```

顶部图不能伪装为背面图。

多视角示例：

```bash
--image-front /path/front.png \
--image-right /path/right.png \
--image-back /path/back.png \
--multiview \
--texture
```

`--texture` 会加载 Hunyuan Paint。若用户明确需要纹理，Agent 不得擅自删除该参数。

当 FitOBJ 使用已有带纹理 GLB 且带 `--postprocess-only` 时，不需要再次写 `--texture`；纹理会从 `SOURCE_GLB` 重新烘焙到 paper model。

---

# 6. Stage 选择规则

## 6.1 图片 → Hunyuan3D → PartField → FitOBJ

必须包含：

```bash
--image-front ...
--image-right ...
--multiview
--texture
--hunyuan-repo ...
--partfield-repo ...
--checkpoint ...
```

## 6.2 已有 GLB → PartField → FitOBJ

第一个参数直接传 GLB：

```bash
python -m partfield_mc.cli "$SOURCE_GLB" \
  --partfield-repo "$PARTFIELD_REPO" \
  --checkpoint "$CHECKPOINT" \
  ...
```

此时不要使用 `--postprocess-only`。

## 6.3 已有 GLB + PLY + NPY → 只运行 FitOBJ

```bash
python -m partfield_mc.cli "$SOURCE_GLB" \
  --postprocess-only \
  --normalized-mesh "$NORMALIZED_MESH" \
  --labels "$LABELS" \
  ...
```

此时不要再传：

```text
--image-*
--multiview
--texture
--hunyuan-repo
```

---

# 7. `--clusters` 规则

## 7.1 特殊值 `1`

```bash
--clusters 1
```

会跳过 PartField，并输出一个全局 AABB。即使同时写：

```bash
--fit-mode primitive
```

也不会产生 Primitive。

## 7.2 一般建议

| 对象 | 初始建议 |
|---|---:|
| 单一方盒 | `1` |
| 简单瓶子：瓶身 + 瓶颈/瓶盖 | `2–4` |
| 苹果：主体 + 梗 + 叶 | `3–4` |
| 简单动物 | `6–10` |
| Puppy / Fox | `8` 起步 |
| 人体/角色 | `8–14` |

聚类数不是语义部件数。结果必须以彩色 PLY 为准。

---

# 8. Fit Mode 总览

V33 支持：

```text
obb
aabb
shared
primitive
```

## 8.1 决策表

| 目标 | 选择 |
|---|---|
| 一个全局方盒 | `clusters=1` |
| Minecraft / 积木 / 最容易制作 | `aabb` |
| 所有方块共享同一倾斜方向 | `shared` |
| 每个分块分别按 PCA 旋转的盒子预览 | `obb` |
| 瓶子、动物、人物、产品等多类型低面形体 | `primitive` |
| 最稳定地避免 Primitive 塌陷 | `primitive + template-mode regular` |
| 需要旧版 constrained surface / frozen interface | `primitive + template-mode adaptive` |

## 8.2 Fit Mode 与 OBJ Mode 兼容矩阵

| fit-mode | merged | separate | surface | all |
|---|---:|---:|---:|---:|
| `obb` | ✓ | ✓ | ✗ | ✗ |
| `aabb` | ✓ | ✓ | ✓ | ✓ |
| `shared` | ✓ | ✓ | ✓ | ✓ |
| `primitive` | ✓ | ✓ | ✓ | ✓ |

`obb + surface/all` 会被拒绝，因为 OBB 无法进入当前严格非重叠 cuboid surface 流程。

---

# 9. `--fit-mode obb`

每个 PartField cluster 独立计算 PCA 方向：

```text
cluster A → 自己的旋转盒
cluster B → 自己的旋转盒
cluster C → 自己的旋转盒
```

适合：

- 查看各 segment 的主方向；
- 机械或长条零件；
- 不要求最终 Paper Model 接触关系。

不适合：

- 标准 `paper_model.obj`；
- 严格非重叠闭合方块装配；
- 有机形状保真。

命令：

```bash
python -m partfield_mc.cli "$SOURCE_GLB" \
  -o obb_result \
  --postprocess-only \
  --normalized-mesh "$NORMALIZED_MESH" \
  --labels "$LABELS" \
  --clusters "$CLUSTERS" \
  --fit-mode obb \
  --obj-mode merged \
  --category generic \
  --up-axis y \
  --forward-axis auto \
  --min-area-ratio 0 \
  --min-faces 4 \
  --grid-divisions 0 \
  --face-resolution 64 \
  --surface-samples 500000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 1
```

---

# 10. `--fit-mode aabb`

每个 cluster 拟合成世界轴对齐长方体：

```text
rotation = identity
只改变中心和 X/Y/Z 尺寸
```

优点：

- 最稳定；
- 不会因 PCA 方向错误而旋转；
- 每块固定 6 个面；
- 适合方块纸模和算法对照基线。

缺点：

- 圆形、倾斜和有机轮廓会被方块化；
- 长斜肢体可能包围较多空白。

Paper Model 推荐：

```bash
--fit-mode aabb \
--obj-mode surface \
--surface-fit-strategy refit
```

## 10.1 `surface-fit-strategy`

### `refit`：推荐

根据每个 cluster 的源表面重新搜索非重叠 AABB，并优先保持源标签邻接。

```bash
--surface-fit-strategy refit
```

### `trim`：旧流程

先拟合盒子，再裁切重叠。

```bash
--surface-fit-strategy trim
```

只用于复现旧结果或诊断。

## 10.2 `semantic-refit`

```text
off
auto
animal
person
```

| 值 | 场景 |
|---|---|
| `off` | 产品、水果、家具、通用物体 |
| `auto` | 允许几何启发式自动判断 |
| `animal` | 狗、狐狸、猫等动物 |
| `person` | 人体或人形角色 |

它不是 LLM Agent，而是确定性几何优先级。

## 10.3 自适应拆分

允许复杂躯干增加少量 cuboid：

```bash
--max-extra-cuboids 1 \
--protected-min-coverage 0.85 \
--split-min-coverage-gain 0.05
```

禁止增加方块：

```bash
--no-adaptive-split
```

## 10.4 通用 AABB Paper Model

```bash
python -m partfield_mc.cli "$SOURCE_GLB" \
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

---

# 11. `--fit-mode shared`

先计算一个模型级公共旋转，所有 cluster 共享该旋转，再分别拟合盒子：

```text
模型公共方向 R
part A → R + 自己的中心和尺寸
part B → R + 自己的中心和尺寸
```

适合：

- 整个模型相对世界轴倾斜；
- 建筑、家具、交通工具；
- 多数部件方向基本一致；
- 希望比 AABB 更贴合但保持统一方向。

不适合：

- 腿、尾巴、耳朵方向差异很大的动物；
- 圆润有机主体。

命令与 AABB 类似，只需改：

```bash
--fit-mode shared
```

---

# 12. `--obj-mode`

## 12.1 `merged`

```bash
--obj-mode merged
```

- 所有部件写入一个 OBJ；
- 适合普通预览；
- Cuboid 模式允许相互穿插；
- Primitive 会输出所有闭合 shell。

## 12.2 `separate`

```bash
--obj-mode separate
```

- 在 `obj_parts/` 中输出每个部件独立 OBJ；
- 适合逐块检查和编辑。

## 12.3 `surface`

```bash
--obj-mode surface
```

Paper Model 推荐模式：

- AABB/shared：执行严格非重叠 refit，并生成 `paper_model.obj`；
- Primitive：输出闭合 primitive shell、connector 和 `paper_model.obj`。

## 12.4 `all`

```bash
--obj-mode all
```

输出 merged、surface、separate 和 paper model 结果。适合算法调试，但处理和文件数量更多。

---

# 13. `--fit-mode primitive`：V33 核心

Primitive 会针对每个最终组生成候选并评分：

```text
box
prism
frustum
cone
ellipsoid
convex
```

评分综合：

- 源表面到候选距离；
- 候选表面到源表面距离；
- 面数与目标面数差；
- 体积和尺寸差；
- 纸模复杂度；
- V33 regular 模式中的规整性惩罚。

---

# 14. Primitive 第一层：`--primitive-template-mode`

## 14.1 `regular`：默认推荐

```bash
--primitive-template-mode regular
```

### 真实行为

1. PartField labels 可以根据 `primitive-part-mode` 合并；
2. 合并后的每个组仍拟合成一个完整、闭合、规整模板；
3. 不使用 constrained surface；
4. 不把 source interface polygon 切入模板；
5. 不允许局部顶点拉伸来迁就接口；
6. 接触在后处理阶段通过已有 attachment face 或小 connector 恢复。

运行时可能看到：

```text
[PrimitiveTemplate] regular mode keeps canonical closed templates;
disabled constrained-surface groups=[...]
```

这是正常日志，表示 V33 主动禁止旧版塌陷减面。

### 适合

- 奶瓶、罐子、瓶盖；
- Puppy、Fox、动物；
- 人物和玩具；
- 机械或产品；
- 任何需要规整、稳定、可展开 primitive 的对象。

## 14.2 `adaptive`：显式旧版兼容

```bash
--primitive-template-mode adaptive
```

### 真实行为

- `auto/surface-patch` 可把连续主体识别为 constrained surface；
- 可冻结原始接口；
- 可局部改变几何以保持 source seam；
- 有 hard face cap 和 repair fallback。

### 仅适合

- 用户明确要求保留不规则凹陷或复杂外表面；
- 标准 primitive 无法接受；
- 能接受更高算法风险，并愿意检查每个输出。

Agent 不应因为对象是“有机形状”就自动选择 adaptive。

---

# 15. Primitive 第二层：`--primitive-part-mode`

支持：

```text
auto
closed
surface-patch
```

## 15.1 在 `regular` 模式下的含义

### `closed`

```bash
--primitive-part-mode closed
```

- 一条 PartField label 对应一个规整闭合模板；
- 不合并 label；
- 最适合头、身体、腿、尾巴已经分开的动物。

### `auto`

```bash
--primitive-part-mode auto
```

- 检测宽大、长接缝、面积相近的表面区域；
- 将这些 labels 合成一个拟合组；
- 每个组拟合成一个规整闭合模板；
- **不会进入 constrained surface**。

例如瓶身被 PartField 切成左右两个大区域时，`auto` 可以先合并，然后拟合成一个 frustum/ellipsoid。

### `surface-patch`

```bash
--primitive-part-mode surface-patch
```

- 比 `auto` 更积极地合并大表面区域；
- 在 regular 模式下仍然输出规整闭合模板，而不是 source surface mesh。

## 15.2 在 `adaptive` 模式下的含义

- `closed`：一 label 一 closed primitive；
- `auto`：主体可能使用 constrained simplification，附肢使用 primitive；
- `surface-patch`：更积极地合并主体表面并使用 constrained surface。

## 15.3 选择规则

| 情况 | 选择 |
|---|---|
| 动物各 label 已是独立部件 | `regular + closed` |
| 瓶身或果体被切成多个大表面区域 | `regular + auto` |
| auto 合并不足，但仍希望规整模板 | `regular + surface-patch` |
| 明确要原表面减面 | `adaptive + auto/surface-patch` |

---

# 16. Primitive 第三层：候选类型

```bash
--primitive-types auto
```

或：

```bash
--primitive-types box,prism,frustum,cone,ellipsoid
```

## 16.1 类型用途

| 类型 | 适合 | 不适合 |
|---|---|---|
| `box` | 方形躯干、盒子、块状零件 | 圆润水果 |
| `prism` | 柱状部件、瓶盖、腿、规则截面 | 明显锥度 |
| `frustum` | 瓶身、肩部、肢体、两端尺寸不同 | 球状主体 |
| `cone` | 尾尖、角、果梗、锥形件 | 扁叶片 |
| `ellipsoid` | 头、身体、果体、圆润容器 | 薄片和尖锐零件 |
| `convex` | 不规则但整体凸的区域 | 规整产品、明显凹形主体 |

## 16.2 V33 regular 中的 `convex`

在 `regular` 模式下：

- `convex` 默认是兜底；
- `--primitive-regularity-weight` 会惩罚不规则 convex；
- 若想彻底避免斜切大平面，直接从 `--primitive-types` 中排除 `convex`。

例如动物：

```bash
--primitive-types box,prism,frustum,cone,ellipsoid
```

奶瓶：

```bash
--primitive-types prism,frustum,ellipsoid,cone
```

---

# 17. Primitive 面数与拟合参数

## 17.1 `--primitive-target-faces`

```bash
--primitive-target-faces 0
```

`0` 表示自动根据源 cluster 面数估计目标复杂度。

手工指定：

```bash
--primitive-target-faces 32
```

会让每个 primitive 倾向接近 32 个 paper faces，但仍受候选拓扑限制。

## 17.2 `--primitive-max-faces`

```bash
--primitive-max-faces 48
```

标准 primitive 单部件最大 canonical paper faces。

建议：

| 目标 | 值 |
|---|---:|
| 极简纸模 | `24–32` |
| 普通动物/产品 | `48–64` |
| 更圆润头部或瓶身 | `64–96` |

## 17.3 `--primitive-max-sides`

限制 prism/frustum/cone/ellipsoid 的 ring side 数：

```bash
--primitive-max-sides 24
```

## 17.4 `--primitive-fit-samples`

每个 cluster 和候选用于评分的表面样本数：

```bash
--primitive-fit-samples 2500
```

建议：

- 快速：`1500–2500`；
- 普通：`2500–4000`；
- 大型光滑产品：`4000–6000`。

## 17.5 `--primitive-complexity-weight`

面数复杂度惩罚：

- 值越高：更偏向简单模板；
- 值越低：更偏向拟合误差。

建议：

```text
动物：0.018–0.030
产品：0.012–0.020
```

## 17.6 `--primitive-regularity-weight`

V33 regular 模式中，对不规则 source-support convex 的额外惩罚：

```bash
--primitive-regularity-weight 0.10
```

- 更规整：提高到 `0.12–0.20`；
- 允许不规则 convex：降低到 `0.03–0.08`。

---

# 18. V33 canonical attachment 规则

V33 不允许任意选一张面切接口，而是为不同模板规定 attachment 区域：

| Primitive | attachment face |
|---|---|
| `box` | 任意已有平面 |
| `prism` | 轴向两端端盖 |
| `frustum` | 轴向两端端盖 |
| `cone` | 底部端盖 |
| `ellipsoid` | 朝向连接方向的极区小面带 |
| `convex` | 所有面，兜底 |

这可以避免：

- 奶瓶侧面出现大圆盘；
- 狗腿从错误侧面连接；
- 锥体尖端被切开；
- 模板被强制斜切。

---

# 19. Primitive 接触模式

支持：

```text
auto
fixed
connector
move
```

## 19.1 `auto`：默认推荐

```bash
--primitive-contact-mode auto
```

每条 source contact 自动分类：

```text
strong
medium
weak
```

### 在 V33 regular 中

```text
strong → existing attachment face 或 bounded connector
medium + connector → bounded connector
medium + separate → 允许分离
weak → 允许分离；若发生穿插，尝试分开弱连接部件
```

注意：regular 中的 strong **不会**把 frozen interface polygon 切入主体。

## 19.2 `fixed`

```bash
--primitive-contact-mode fixed
```

### regular 模式

`fixed` 的含义已改变：

- 请求所有 source contacts 都连接；
- 但仍然保持模板形状；
- 使用 direct canonical face 或 bounded connector；
- 不执行旧版 frozen-interface cutting。

### adaptive 模式

会尝试重建和冻结精确 source interface。

## 19.3 `connector`

```bash
--primitive-contact-mode connector
```

- 主部件不移动；
- 每条相邻关系使用 connector；
- 适合全部部件都需要装配关系，但不要求共享原始接口形状。

## 19.4 `move`

```bash
--primitive-contact-mode move
```

旧模式，通过移动或旋转子部件恢复接触。可能改变姿态，Agent 默认禁止选择。

---

# 20. 接触强度自动分类

## 20.1 核心参数

```bash
--primitive-contact-weak-threshold 0.20 \
--primitive-contact-strong-threshold 0.55 \
--primitive-contact-min-edge-count 6 \
--primitive-contact-medium-mode connector
```

必须满足：

```text
0 <= weak_threshold < strong_threshold <= 1
```

接触边数少于 `min-edge-count` 时，会优先视为 weak，避免把点接触或极短边接触强制扩大。

## 20.2 预设

### 动物装配

```bash
--primitive-contact-weak-threshold 0.18 \
--primitive-contact-strong-threshold 0.50 \
--primitive-contact-min-edge-count 6 \
--primitive-contact-medium-mode connector
```

### 产品/瓶盖：更容易恢复视觉接触

```bash
--primitive-contact-weak-threshold 0.12 \
--primitive-contact-strong-threshold 0.42 \
--primitive-contact-min-edge-count 6 \
--primitive-contact-medium-mode connector
```

### 枝叶允许分离

```bash
--primitive-contact-weak-threshold 0.28 \
--primitive-contact-strong-threshold 0.62 \
--primitive-contact-min-edge-count 8 \
--primitive-contact-medium-mode separate
```

---

# 21. V33 空间近接触推断

Hunyuan3D 可能生成视觉上接触，但拓扑上存在小缝的部件，例如：

```text
瓶盖 ↔ 瓶颈
腿 ↔ 身体
耳朵 ↔ 头
尾巴 ↔ 躯干
```

如果只看共享边，会得到：

```text
edge_count = 0
```

V33 使用空间距离补充接触：

```bash
--primitive-contact-proximity-ratio 0.015 \
--primitive-contact-proximity-min-points 8 \
--primitive-contact-proximity-min-coverage 0.01
```

含义：

- `proximity-ratio`：最大近接触距离，相对模型最长边；
- `min-points`：至少多少个源点靠近；
- `min-coverage`：近接触点至少占较小部件采样点的比例。

建议：

| 场景 | proximity ratio |
|---|---:|
| 已经拓扑相接 | `0.005–0.015` |
| 普通动物 | `0.015–0.020` |
| 瓶盖存在明显小缝 | `0.020–0.030` |
| 完全不希望推断空间接触 | `0` |

比例过大可能把相邻但不应连接的部件误判为接触。

---

# 22. Connector 参数

```bash
--primitive-connector-sides 4 \
--primitive-connector-radius-ratio 0.018 \
--primitive-connector-inset-ratio 0.35 \
--primitive-connector-min-length-ratio 0.002
```

| 参数 | 含义 |
|---|---|
| `connector-sides` | 横截面边数，4 最容易制作；圆形产品可用 6–8 |
| `radius-ratio` | 相对模型最长边的首选半径 |
| `inset-ratio` | 接触中心向 attachment face 内部移动，防止边角接触 |
| `min-length-ratio` | 面很近时的最小 connector 长度 |

## 22.1 防止大白色法兰

V33 新增：

```bash
--primitive-regular-connector-max-face-area-ratio 0.08
```

connector 端面最多占较小 attachment face 的比例。

建议：

| 场景 | 值 |
|---|---:|
| 奶瓶、瓶盖 | `0.03–0.05` |
| Puppy / Fox | `0.05–0.08` |
| 需要更强装配 | `0.08–0.12` |

出现大白色圆盘或法兰时，应先减小该参数，而不是放大 `primitive-interface-*`。

---

# 23. Interface 参数在 V33 中的作用

```bash
--primitive-interface-max-sides 8 \
--primitive-interface-min-width-ratio 0.006 \
--primitive-interface-plane-tolerance-ratio 0.000001
```

## regular 模式

- 用于 source seam 估计、分组和接触强度诊断；
- 不会把重建的 interface polygon 切入模板；
- 不应作为修复模板形状的主要参数。

## adaptive 模式

- 可控制冻结接口多边形；
- `max-sides` 影响接口复杂度；
- `min-width-ratio` 处理退化线/点接口；
- `plane-tolerance-ratio` 仅是数值容差，不应持续放大来掩盖拓扑错误。

---

# 24. Validation policy

```text
repair
warn
strict
```

## `repair`：默认

```bash
--primitive-validation-policy repair
```

- 尝试修复；
- 普通几何歧义记录警告并继续；
- regular 模板接触后检查 shape signature；
- 适合正式批量运行。

## `warn`

```bash
--primitive-validation-policy warn
```

尽量不终止，适合探索，但必须检查 JSON 和模型。

## `strict`

```bash
--primitive-validation-policy strict
```

任何关键验证失败立即终止。只用于测试和算法调试。

---

# 25. Primitive 重叠处理

默认：

```text
primitive_resolve_overlaps = true
```

可关闭：

```bash
--no-primitive-resolve-overlaps
```

接触自动分类时：

- strong 和 medium connector 关系受到保护；
- weak 关系允许分离；
- 非邻接部件若穿插，程序会尝试缩放/平移较小部件；
- regular 模式仍会检查模板形状签名。

如果模型姿态必须完全保持，且愿意手动处理穿插，可关闭自动重叠处理。

---

# 26. 其他高级开关

## 26.1 完全关闭 Primitive 接触恢复

```bash
--no-primitive-preserve-contacts
```

使用后，程序不会根据 PartField 邻接、接触强度或空间近接触生成 direct joint/connector。只适合需要所有部件完全独立的诊断任务。

## 26.2 关闭非邻接 Primitive 重叠修复

```bash
--no-primitive-resolve-overlaps
```

用于保持原始拟合位置，但可能留下非邻接部件穿插。

## 26.3 隐藏插入深度

```bash
--primitive-contact-overlap-ratio 0
```

默认 `0` 表示不额外插入。正值会让连接件或接触面产生隐藏穿入，不适合要求精确共面纸模时使用。

## 26.4 强制重新运行 PartField

```bash
--force
```

当缓存损坏、输入模型发生变化或希望清理旧 PartField 结果时使用。若只是修改 FitOBJ 参数，应使用 `--postprocess-only`，而不是 `--force`。

## 26.5 Category

CLI 支持：

```text
generic
auto
animal
person
```

- `generic`：产品、水果、家具；
- `auto`：允许几何启发式分类；
- `animal`：动物语义命名和 AABB refit 优先级；
- `person`：人体语义命名和 AABB refit 优先级。

---

# 27. Agent 自动选参决策树

Agent 必须按以下顺序决策。

## STEP 1：判断运行阶段

```text
只有图片：
    Hunyuan3D + PartField + FitOBJ
    保留 --texture

已有带纹理 GLB，没有 PLY/NPY：
    PartField + FitOBJ

已有 GLB + normalized mesh + labels：
    --postprocess-only
```

## STEP 2：验证路径

```text
SOURCE_GLB 必须存在
NORMALIZED_MESH 必须是 input_UID_0.ply
LABELS 必须是 UID_0_NN.npy
NN 必须等于 --clusters
```

若任何路径缺失，Agent 必须先给查找命令，不得猜测。

## STEP 3：选择 fit-mode

```text
单盒 → clusters=1
方块纸模 → aabb
统一倾斜方块 → shared
独立方向盒子预览 → obb
其他低面有机/产品形状 → primitive
```

## STEP 4：Primitive template mode

```text
默认 → regular
明确要求复杂原始表面/冻结接口 → adaptive
```

## STEP 5：Primitive part mode

```text
动物独立头身腿尾 → closed
瓶身/果体被切为大表面区域 → auto
需要更积极合并 → surface-patch
```

注意 regular 下 auto/surface-patch 只合并 labels，最后仍拟合规整模板。

## STEP 6：Primitive types

```text
未知对象 → auto
奶瓶 → prism,frustum,ellipsoid,cone
动物 → box,prism,frustum,cone,ellipsoid
水果主体 → ellipsoid,frustum,prism,cone
避免大斜切 → 排除 convex
```

## STEP 7：Contact

```text
一般 → auto
所有关系都必须连接但模板不能变形 → regular + fixed
所有相邻关系都加小连接件 → connector
旧移动兼容 → move
```

## STEP 8：Proximity

```text
瓶盖/肢体存在小缝 → 0.020–0.030
普通动物 → 0.015–0.020
禁止空间推断 → 0
```

## STEP 9：面数和纹理

```text
standard primitive max faces：48–64
更圆润：72–96
face-resolution：
    总面数 <= 100 → 96–128
    100–300 → 64–96
    300–800 → 48–64
```

---

# 28. Agent 输出格式

建议先输出结构化判断，再输出完整命令：

```json
{
  "version": "V33",
  "run_stage": "fitobj_only",
  "fit_mode": "primitive",
  "obj_mode": "surface",
  "template_mode": "regular",
  "part_mode": "closed",
  "reason": "animal parts should remain independent regular closed shells",
  "paths": {
    "source_glb": "/absolute/path/hunyuan3d_textured_multiview.glb",
    "normalized_mesh": "/absolute/path/input_UID_0.ply",
    "labels": "/absolute/path/UID_0_08.npy"
  },
  "parameters": {
    "clusters": 8,
    "primitive_types": "box,prism,frustum,cone,ellipsoid",
    "primitive_contact_mode": "auto",
    "primitive_contact_proximity_ratio": 0.015,
    "primitive_regular_connector_max_face_area_ratio": 0.06
  },
  "requires_partfield_rerun": false,
  "command": "python -m partfield_mc.cli ..."
}
```

Agent 禁止：

- 猜测不存在的文件路径；
- 把彩色 segmentation PLY 当 normalized mesh；
- 把工作目录 GLB 当 normalized mesh；
- 把 `_03.npy` 配成 `--clusters 8`；
- 在 OBB 下使用 `surface/all`；
- 声称 V33 加载外部 OBJ/GLB preset library；
- 在 regular 模式中通过 `primitive-surface-*` 调整主体形状；
- 因 Fit-only 命令没有 `--texture` 就声称纹理被取消。

---

# 29. 完整命令：图片 → Hunyuan3D → PartField → AABB

```bash
python -m partfield_mc.cli \
  --image-front /absolute/path/front.png \
  --image-right /absolute/path/right.png \
  --image-back /absolute/path/back.png \
  --multiview \
  --texture \
  --hunyuan-repo /mnt/e/yp/Hunyuan3D-2 \
  --partfield-repo /mnt/e/yp/PartField \
  --checkpoint /mnt/e/yp/PartField/model/model_objaverse.ckpt \
  --simplify-faces 5000 \
  --n-point-per-face 50 \
  --n-sample-each 1000 \
  -o full_aabb_result \
  --clusters 8 \
  --clustering agglo \
  --adjacency mst \
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
  --face-resolution 96 \
  --surface-samples 500000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 2
```

---

# 30. 完整命令：已有 GLB → PartField → V33 Regular Primitive

```bash
python -m partfield_mc.cli \
  "$SOURCE_GLB" \
  --partfield-repo /mnt/e/yp/PartField \
  --checkpoint /mnt/e/yp/PartField/model/model_objaverse.ckpt \
  --simplify-faces 5000 \
  --n-point-per-face 50 \
  --n-sample-each 1000 \
  -o primitive_result \
  --clusters 8 \
  --clustering agglo \
  --adjacency mst \
  --category animal \
  --up-axis y \
  --forward-axis auto \
  --fit-mode primitive \
  --obj-mode surface \
  --primitive-template-mode regular \
  --primitive-part-mode closed \
  --primitive-types box,prism,frustum,cone,ellipsoid \
  --primitive-target-faces 0 \
  --primitive-max-faces 64 \
  --primitive-max-sides 24 \
  --primitive-fit-samples 3000 \
  --primitive-complexity-weight 0.020 \
  --primitive-regularity-weight 0.10 \
  --primitive-contact-mode auto \
  --primitive-contact-proximity-ratio 0.015 \
  --primitive-contact-proximity-min-points 6 \
  --primitive-contact-proximity-min-coverage 0.008 \
  --primitive-contact-weak-threshold 0.18 \
  --primitive-contact-strong-threshold 0.50 \
  --primitive-contact-min-edge-count 6 \
  --primitive-contact-medium-mode connector \
  --primitive-connector-sides 4 \
  --primitive-connector-radius-ratio 0.018 \
  --primitive-connector-inset-ratio 0.35 \
  --primitive-connector-min-length-ratio 0.002 \
  --primitive-regular-connector-max-face-area-ratio 0.06 \
  --primitive-validation-policy repair \
  --face-resolution 64 \
  --surface-samples 500000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 2
```

---

# 31. 完整命令：只运行 FitOBJ Regular Primitive

```bash
python -m partfield_mc.cli \
  "$SOURCE_GLB" \
  -o primitive_fitonly_v33 \
  --postprocess-only \
  --normalized-mesh "$NORMALIZED_MESH" \
  --labels "$LABELS" \
  --clusters "$CLUSTERS" \
  --category generic \
  --up-axis y \
  --forward-axis auto \
  --fit-mode primitive \
  --obj-mode surface \
  --primitive-template-mode regular \
  --primitive-part-mode auto \
  --primitive-types auto \
  --primitive-target-faces 0 \
  --primitive-max-faces 64 \
  --primitive-max-sides 24 \
  --primitive-fit-samples 3000 \
  --primitive-complexity-weight 0.020 \
  --primitive-regularity-weight 0.10 \
  --primitive-contact-mode auto \
  --primitive-contact-proximity-ratio 0.015 \
  --primitive-contact-proximity-min-points 8 \
  --primitive-contact-proximity-min-coverage 0.01 \
  --primitive-contact-weak-threshold 0.20 \
  --primitive-contact-strong-threshold 0.55 \
  --primitive-contact-min-edge-count 6 \
  --primitive-contact-medium-mode connector \
  --primitive-connector-sides 4 \
  --primitive-connector-radius-ratio 0.020 \
  --primitive-connector-inset-ratio 0.35 \
  --primitive-regular-connector-max-face-area-ratio 0.06 \
  --primitive-validation-policy repair \
  --face-resolution 64 \
  --surface-samples 500000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 2
```

---

# 32. 奶瓶 V33 Regular 推荐参数

适用特点：

- 瓶身接近 frustum/ellipsoid；
- 瓶盖适合 prism/frustum；
- 瓶盖和瓶颈可能有微小拓扑缝隙；
- 禁止出现大白色法兰。

```bash
python -m partfield_mc.cli \
  "$SOURCE_GLB" \
  -o milk_primitive_v33_regular \
  --postprocess-only \
  --normalized-mesh "$NORMALIZED_MESH" \
  --labels "$LABELS" \
  --clusters 3 \
  --category generic \
  --up-axis y \
  --forward-axis auto \
  --fit-mode primitive \
  --obj-mode surface \
  --primitive-template-mode regular \
  --primitive-part-mode auto \
  --primitive-types prism,frustum,ellipsoid,cone \
  --primitive-target-faces 0 \
  --primitive-max-faces 64 \
  --primitive-max-sides 24 \
  --primitive-fit-samples 4000 \
  --primitive-complexity-weight 0.015 \
  --primitive-regularity-weight 0.10 \
  --primitive-contact-mode auto \
  --primitive-contact-proximity-ratio 0.025 \
  --primitive-contact-proximity-min-points 8 \
  --primitive-contact-proximity-min-coverage 0.01 \
  --primitive-contact-weak-threshold 0.12 \
  --primitive-contact-strong-threshold 0.42 \
  --primitive-contact-min-edge-count 6 \
  --primitive-contact-medium-mode connector \
  --primitive-connector-sides 8 \
  --primitive-connector-radius-ratio 0.018 \
  --primitive-connector-inset-ratio 0.35 \
  --primitive-connector-min-length-ratio 0.002 \
  --primitive-regular-connector-max-face-area-ratio 0.05 \
  --primitive-validation-policy repair \
  --face-resolution 128 \
  --surface-samples 750000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 3
```

如果瓶盖仍被判为 weak：

1. 确认 PartField 确实分出瓶盖；
2. 把 proximity ratio 从 `0.025` 提高到 `0.030`；
3. 将 weak threshold 降至 `0.08–0.10`；
4. 不要直接使用 `move`。

如果出现大接头：

```bash
--primitive-regular-connector-max-face-area-ratio 0.03
```

---

# 33. Puppy / Fox V33 Regular 推荐参数

适用特点：

- 头、身体、腿、耳朵、尾巴应保持独立闭合 shell；
- 不允许 constrained surface 塌陷；
- 中等连接用小 connector；
- 很弱的耳尖/尾尖接触允许分离。

```bash
python -m partfield_mc.cli \
  "$SOURCE_GLB" \
  -o puppy_primitive_v33_regular \
  --postprocess-only \
  --normalized-mesh "$NORMALIZED_MESH" \
  --labels "$LABELS" \
  --clusters 8 \
  --category animal \
  --up-axis y \
  --forward-axis auto \
  --fit-mode primitive \
  --obj-mode surface \
  --primitive-template-mode regular \
  --primitive-part-mode closed \
  --primitive-types box,prism,frustum,cone,ellipsoid \
  --primitive-target-faces 0 \
  --primitive-max-faces 64 \
  --primitive-max-sides 24 \
  --primitive-fit-samples 3000 \
  --primitive-complexity-weight 0.020 \
  --primitive-regularity-weight 0.10 \
  --primitive-contact-mode auto \
  --primitive-contact-proximity-ratio 0.015 \
  --primitive-contact-proximity-min-points 6 \
  --primitive-contact-proximity-min-coverage 0.008 \
  --primitive-contact-weak-threshold 0.18 \
  --primitive-contact-strong-threshold 0.50 \
  --primitive-contact-min-edge-count 6 \
  --primitive-contact-medium-mode connector \
  --primitive-connector-sides 4 \
  --primitive-connector-radius-ratio 0.018 \
  --primitive-connector-inset-ratio 0.35 \
  --primitive-regular-connector-max-face-area-ratio 0.06 \
  --primitive-validation-policy repair \
  --face-resolution 64 \
  --surface-samples 500000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 2
```

若多个 labels 只是同一个躯干表面分区，可测试：

```bash
--primitive-part-mode auto
```

但必须先查看 segmented PLY。

---

# 34. 苹果/水果在 V33 的选择

## 33.1 稳定规整纸模优先

```bash
--primitive-template-mode regular \
--primitive-part-mode auto \
--primitive-types ellipsoid,frustum,prism,cone \
--primitive-contact-mode auto \
--primitive-contact-medium-mode separate
```

效果：

- 果体表面 labels 可先合并；
- 果体拟合为一个规整 ellipsoid/frustum；
- 枝叶弱连接可分离；
- 不会进入三万面 constrained surface。

代价：苹果顶部凹陷、底部不规则轮廓可能被理想化。

## 33.2 原始复杂轮廓优先

只有用户明确接受风险时：

```bash
--primitive-template-mode adaptive \
--primitive-part-mode auto \
--primitive-surface-hard-max-faces 256 \
--primitive-validation-policy repair
```

运行后必须检查：

```text
selected=constrained_surface
paper_faces
budget_ok
mandatory_reduction_satisfied
```

---

# 35. Adaptive 专用参数

以下参数仅在 `--primitive-template-mode adaptive` 下对 constrained surface 有实际意义：

```bash
--primitive-surface-main-body-min-area-ratio 0.35 \
--primitive-surface-boundary-rings 0 \
--primitive-surface-search-steps 18 \
--primitive-surface-min-reduction-ratio 0.15 \
--primitive-surface-hard-max-faces 256
```

在 `regular` 模式中，Agent 不应把它们当作调整主体形状的手段。

---

# 36. 纹理参数

## 35.1 `--face-resolution`

每个 paper face 的纹理 tile 分辨率。

建议：

| 总 paper faces | 建议 |
|---|---:|
| ≤ 100 | `96–128` |
| 101–300 | `64–96` |
| 301–800 | `48–64` |
| > 800 | 先降低面数，不建议直接贴图 |

## 35.2 `--surface-samples`

```bash
--surface-samples 500000
```

- 普通动物：`500000`；
- 有包装文字的产品：`750000–1000000`；
- 低内存调试：`150000–300000`。

## 35.3 其他参数

```bash
--palette-size 0 \
--texture-filter bilinear \
--uv-wrap clamp \
--padding 2
```

`palette-size 0` 表示不做调色板量化。

---

# 37. 输出文件

## 36.1 通用输出

```text
mc_model.glb
mc_model.obj
mc_model.mtl
mc_texture.png
parts.json
partfield_segmented.ply
```

## 36.2 Paper Model 输出

使用 `--obj-mode surface/all`：

```text
paper_model.obj
paper_model.mtl
paper_model_texture.png
paper_model.glb
paper_model_parts.json
```

Blender Paper Model 推荐导入：

```text
paper_model.obj
```

## 36.3 AABB surface 诊断输出

```text
mc_model_raw_aabb.glb
mc_model_before_surface.glb
mc_model_after_surface.glb
parts_raw_aabb.json
parts_before_surface.json
parts_after_surface.json
```

---

# 38. Primitive JSON 重点字段

在 `paper_model_parts.json` 或 `parts.json` 中检查：

```text
primitive_type
paper_face_count
triangle_count
fit_score
source_segment_ids
primitive_template_mode
canonical_template_topology_locked
local_interface_deformation_applied
regular_template_shape_preserved_after_contacts
contact_strength_classification
allowed_separated_contact_edges
contact_required_graph_connected
spatially_connected_assembly
connector_count
```

## regular 正常状态

```json
{
  "primitive_template_mode": "regular",
  "canonical_template_topology_locked": true,
  "local_interface_deformation_applied": false,
  "regular_template_shape_preserved_after_contacts": true
}
```

## 接触状态

```text
contact_required_graph_connected = true
```

表示所有必须连接的 strong/medium 关系均满足。

```text
spatially_connected_assembly = false
```

不一定是错误，可能是 weak 或 medium separate 被允许分离。

---

# 39. 运行日志解释

## 38.1 Regular 主动禁用 constrained surface

```text
[PrimitiveTemplate] regular mode keeps canonical closed templates;
disabled constrained-surface groups=[...]
```

正常。

## 38.2 选择候选

```text
[PrimitiveFit] segment=... selected=frustum_8 paper_faces=...
```

说明选择了规整 frustum。

若大量部件全部选择 ellipsoid：

- 检查 PartField 分割；
- 限制 `primitive-types`；
- 提高 `primitive-regularity-weight` 不能解决类型单一问题，应明确排除类型。

## 38.3 空间接触

JSON 中 `source_kind=spatial_proximity` 表示通过近距离表面推断接触。

## 38.4 Texture 总面数

```text
[PrimitiveTexture] 1/NNN
```

`NNN` 是所有 source parts 和 connectors 的 paper face 总数。

regular 模式通常应是几十到几百，不应出现数万。

---

# 40. 常见错误与修复

## 39.1 `--postprocess-only requires --normalized-mesh and --labels`

原因：变量为空或未传参数。

```bash
printf 'SOURCE_GLB=[%s]\n' "$SOURCE_GLB"
printf 'NORMALIZED_MESH=[%s]\n' "$NORMALIZED_MESH"
printf 'LABELS=[%s]\n' "$LABELS"
```

## 39.2 `Label count does not match face count`

原因：normalized mesh 与 labels 不匹配。

最常见错误是使用：

```text
partfield_clusters/.../ply/*_0_NN.ply
```

代替：

```text
PartField/exp_results/.../input_UID_0.ply
```

## 39.3 模型塌陷

检查是否使用：

```bash
--primitive-template-mode adaptive
```

修复优先级：

1. 改为 `regular`；
2. 动物使用 `--primitive-part-mode closed`；
3. 排除 `convex`；
4. 不使用 constrained surface 参数。

## 39.4 出现大白色法兰/圆盘

```bash
--primitive-regular-connector-max-face-area-ratio 0.03
--primitive-connector-radius-ratio 0.012
```

同时确认正在使用 `regular`。

## 39.5 瓶盖漂浮

1. 检查 PartField 是否独立分出瓶盖；
2. 增大：

```bash
--primitive-contact-proximity-ratio 0.025
```

3. 降低 weak threshold；
4. medium mode 使用 connector。

## 39.6 腿、耳朵、尾巴被错误连接

- 降低 proximity ratio；
- 提高 weak threshold；
- 提高 min edge count；
- 或 medium mode 改为 separate。

## 39.7 Hunyuan Paint `Cannot allocate memory`

这是系统内存/swap 峰值问题，不是 FitOBJ 参数问题。

用户需要纹理时不能删除 `--texture`。应：

- 增加 WSL memory/swap；
- 关闭其他进程；
- 或将 Shape 和 Paint 分为不同 Python 进程。

## 39.8 shell 中出现奇怪的 `>` 提示

这是 Bash 多行输入提示符。确保每行末尾的 `\` 后没有多余字符，并重新复制完整命令。

---

# 41. Agent 运行前检查清单

```text
[ ] 已确定运行阶段
[ ] SOURCE_GLB 是原始带纹理 Hunyuan GLB
[ ] NORMALIZED_MESH 是 input_UID_0.ply
[ ] LABELS 是 UID_0_NN.npy
[ ] NN 与 clusters 一致
[ ] segmented PLY 已检查
[ ] fit-mode 与 obj-mode 兼容
[ ] Primitive 默认 template-mode regular
[ ] 动物默认 part-mode closed
[ ] 产品连续主体默认 part-mode auto
[ ] proximity 参数没有过大
[ ] connector 面积上限已设置
[ ] 输出目录与旧结果分开
```

---

# 42. Agent 运行后检查清单

```text
[ ] 控制台没有终止错误
[ ] regular 模式没有 selected=constrained_surface
[ ] paper face 总数合理
[ ] primitive 类型符合部件形状
[ ] 无大白色法兰
[ ] 无局部塌陷或尖刺
[ ] regular_template_shape_preserved_after_contacts=true
[ ] contact_required_graph_connected=true 或有合理 warning
[ ] allowed_separated_contact_edges 符合预期
[ ] paper_model.obj 可在 Blender 打开
[ ] paper_model_texture.png 正常
```

---

# 43. 最终推荐默认值

## Cuboid Paper Model

```bash
--fit-mode aabb \
--obj-mode surface \
--surface-fit-strategy refit \
--refit-min-coverage 0.05 \
--refit-beam-width 128 \
--part-gap-ratio 0
```

## V33 Regular Organic/Product Paper Model

```bash
--fit-mode primitive \
--obj-mode surface \
--primitive-template-mode regular \
--primitive-part-mode auto \
--primitive-types box,prism,frustum,cone,ellipsoid \
--primitive-max-faces 64 \
--primitive-max-sides 24 \
--primitive-contact-mode auto \
--primitive-contact-proximity-ratio 0.015 \
--primitive-contact-weak-threshold 0.20 \
--primitive-contact-strong-threshold 0.55 \
--primitive-contact-min-edge-count 6 \
--primitive-contact-medium-mode connector \
--primitive-regular-connector-max-face-area-ratio 0.06 \
--primitive-validation-policy repair
```

## V33 Animal

```bash
--primitive-template-mode regular \
--primitive-part-mode closed \
--primitive-types box,prism,frustum,cone,ellipsoid \
--primitive-contact-proximity-ratio 0.015 \
--primitive-contact-weak-threshold 0.18 \
--primitive-contact-strong-threshold 0.50 \
--primitive-contact-medium-mode connector
```

## V33 Bottle/Product

```bash
--primitive-template-mode regular \
--primitive-part-mode auto \
--primitive-types prism,frustum,ellipsoid,cone \
--primitive-contact-proximity-ratio 0.025 \
--primitive-contact-weak-threshold 0.12 \
--primitive-contact-strong-threshold 0.42 \
--primitive-contact-medium-mode connector \
--primitive-regular-connector-max-face-area-ratio 0.05
```

---

# 44. 一句话总结

```text
方块纸模选 AABB；统一倾斜方块选 Shared；独立旋转盒预览选 OBB；
低面有机和产品部件选 Primitive。V33 Primitive 默认使用 regular，保持规整模板，
通过 canonical attachment face、小 connector 和空间近接触恢复装配关系；
只有明确要求复杂原始曲面时才使用 adaptive。
```
