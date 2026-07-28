# Hunyuan3D → PartField → FitOBJ 分阶段运行说明（V30）

本说明用于将当前流水线拆成三个可独立重复执行的阶段：

```text
多视角/单张图片
    ↓ Stage 1：Hunyuan3D
原始高精度带纹理 GLB
    ↓ Stage 2：PartField
normalized_mesh.ply + labels.npy
    ↓ Stage 3：FitOBJ
AABB 或 Primitive Paper Model
```

拆分后的主要好处是：修改 AABB/Primitive 拟合算法或参数时，只需要重复运行 **Stage 3**，不需要再次运行 Hunyuan3D 和 PartField。

---

## 0. 环境与公共路径

进入项目并安装当前代码：

```bash
conda activate partfield

cd /mnt/e/yp/partfield_hunyuan_v30
python -m pip install -r requirements_wrapper.txt
python -m pip install -e .
```

设置本次狐狸模型使用的公共变量：

```bash
export PROJECT=/mnt/e/yp/partfield_hunyuan_v30
export HUNYUAN_REPO=/mnt/e/yp/Hunyuan3D-2
export PARTFIELD_REPO=/mnt/e/yp/PartField
export CHECKPOINT=/mnt/e/yp/PartField/model/model_objaverse.ckpt

export FRONT_IMAGE=/mnt/e/yp/test/fox_front.png
export SIDE_IMAGE=/mnt/e/yp/test/fox_side.png
export BACK_IMAGE=/mnt/e/yp/test/fox_back.png

export WORK=$PROJECT/fox_stages
export CLUSTERS=8

mkdir -p "$WORK/01_hunyuan" \
         "$WORK/02_partfield" \
         "$WORK/03_fitobj_aabb" \
         "$WORK/03_fitobj_primitive"
```

下面所有路径均为 WSL/Linux 路径。

---

# Stage 1：单独运行 Hunyuan3D

当前 CLI 会在图片生成完成后继续执行 PartField，因此只运行 Hunyuan3D 时，直接调用项目内的 `generate_mesh()` API。

## 1.1 多视角生成带纹理 GLB

```bash
export SOURCE_GLB=$WORK/01_hunyuan/hunyuan3d_textured_multiview.glb

python - <<'PY'
import os
from partfield_mc.hunyuan3d import generate_mesh

generate_mesh(
    image=None,
    images={
        "front": os.environ["FRONT_IMAGE"],
        "left": os.environ["SIDE_IMAGE"],
        "back": os.environ["BACK_IMAGE"],
    },
    output=os.environ["SOURCE_GLB"],
    hunyuan_repo=os.environ["HUNYUAN_REPO"],
    texture=True,
    multiview=True,
)
PY
```

输出：

```text
fox_stages/01_hunyuan/hunyuan3d_textured_multiview.glb
```

检查：

```bash
ls -lh "$SOURCE_GLB"
```

## 1.2 单张图片生成（可选）

```bash
export SINGLE_IMAGE=/mnt/e/yp/test/fox_front.png
export SOURCE_GLB=$WORK/01_hunyuan/hunyuan3d_textured.glb

python - <<'PY'
import os
from partfield_mc.hunyuan3d import generate_mesh

generate_mesh(
    image=os.environ["SINGLE_IMAGE"],
    images=None,
    output=os.environ["SOURCE_GLB"],
    hunyuan_repo=os.environ["HUNYUAN_REPO"],
    texture=True,
    multiview=False,
)
PY
```

后续 Paper Model 推荐使用带纹理的原始 Hunyuan3D GLB。

---

# Stage 2：单独运行 PartField

这一阶段会：

1. 将 Hunyuan3D 高面数模型减面到约 5000 faces；
2. 运行 PartField feature inference；
3. 聚类为指定数量的部件；
4. 输出 FitOBJ 所需的 `normalized_mesh` 和 `labels`；
5. 自动生成一个环境变量文件，供 Stage 3 直接读取。

先确认 Stage 1 的模型路径：

```bash
export SOURCE_GLB=$WORK/01_hunyuan/hunyuan3d_textured_multiview.glb
```

运行 PartField：

```bash
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
    target_faces=5000,
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
    lines.append(
        f"export SEGMENTED_PLY={shlex.quote(str(artifacts.colored_segmentation))}"
    )
env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\nPartField completed")
print(f"PARTFIELD_INPUT_SIMPLIFIED={simplified_glb}")
print(f"NORMALIZED_MESH={artifacts.normalized_mesh}")
print(f"LABELS={artifacts.labels_path}")
print(f"ARTIFACT_ENV={env_file}")
PY
```

加载 PartField 输出路径：

```bash
source "$WORK/02_partfield/partfield_artifacts.env"

printf 'SOURCE_GLB=%s\n' "$SOURCE_GLB"
printf 'NORMALIZED_MESH=%s\n' "$NORMALIZED_MESH"
printf 'LABELS=%s\n' "$LABELS"
```

检查必要文件：

```bash
test -f "$SOURCE_GLB"       && echo "SOURCE_GLB OK"
test -f "$NORMALIZED_MESH"  && echo "NORMALIZED_MESH OK"
test -f "$LABELS"           && echo "LABELS OK"
```

## Stage 2 主要参数

| 参数 | 推荐值 | 作用 |
|---|---:|---|
| `target_faces` | `5000` | 仅减小传给 PartField 的模型；不会替换后续贴图来源 |
| `clusters` | `8` | 请求输出 8 个 PartField 部件 |
| `clustering` | `agglo` | 层次聚类 |
| `adjacency` | `mst` | 使用 MST/KNN 邻接关系 |
| `n_point_per_face` | `50` | Hunyuan3D 模型的推荐采样值 |
| `n_sample_each` | `1000` | Hunyuan3D 模型的推荐采样值 |
| `preprocess_mesh` | `False` | 保持当前稳定配置 |
| `force` | `False` | 复用缓存；设为 `True` 会删除并重跑对应 PartField 结果 |

> `labels.npy` 的聚类数量必须与 Stage 3 的 `--clusters` 一致。例如 `_08.npy` 对应 `--clusters 8`。

---

# Stage 3：单独运行 FitOBJ

先加载 Stage 2 保存的路径：

```bash
cd "$PROJECT"
source "$WORK/02_partfield/partfield_artifacts.env"
```

FitOBJ 的三个关键输入是：

```text
SOURCE_GLB       原始高精度、带纹理 Hunyuan3D 模型
NORMALIZED_MESH  PartField 输出的 input_UID_0.ply
LABELS           PartField 输出的 UID_0_NN.npy
```

**不要**把 `partfield_input_simplified.glb` 作为 FitOBJ 的第一个输入。它只用于分割；贴图烘焙必须使用原始带纹理 `SOURCE_GLB`。

---

# 3A. FitOBJ：AABB Paper Model

AABB 模式将每个 PartField cluster 拟合为与世界坐标轴对齐的闭合长方体。`--obj-mode surface` 会运行非重叠约束重新拟合，并导出标准 Paper Model。

## 3A.1 完整命令

```bash
rm -rf "$WORK/03_fitobj_aabb"

python -m partfield_mc.cli \
  "$SOURCE_GLB" \
  -o "$WORK/03_fitobj_aabb" \
  --postprocess-only \
  --normalized-mesh "$NORMALIZED_MESH" \
  --labels "$LABELS" \
  --clusters "$CLUSTERS" \
  --category animal \
  --up-axis y \
  --fit-mode aabb \
  --obj-mode surface \
  --surface-fit-strategy refit \
  --refit-min-coverage 0.05 \
  --refit-beam-width 64 \
  --semantic-refit animal \
  --max-extra-cuboids 1 \
  --protected-min-coverage 0.85 \
  --split-min-coverage-gain 0.05 \
  --part-gap-ratio 0 \
  --face-resolution 64 \
  --surface-samples 500000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap repeat \
  --padding 1
```

## 3A.2 AABB 关键参数

| 参数 | 推荐值 | 作用 |
|---|---:|---|
| `--fit-mode aabb` | 必需 | 每个 cluster 使用世界坐标轴对齐长方体 |
| `--obj-mode surface` | 必需 | 非重叠约束拟合并输出 `paper_model.*` |
| `--surface-fit-strategy refit` | `refit` | 从源 cluster 重新搜索非重叠 AABB，而不是后裁剪 |
| `--refit-min-coverage` | `0.05` | AABB 至少覆盖的源表面积比例 |
| `--refit-beam-width` | `64` | 非重叠候选搜索宽度；越高越慢，但候选更充分 |
| `--semantic-refit` | `animal` | 使用动物几何优先级，优先保护头/脸等结构 |
| `--max-extra-cuboids` | `1` | 允许自适应拆分额外增加 1 个长方体 |
| `--protected-min-coverage` | `0.85` | 保护重要部位后，躯干覆盖不足时考虑拆分 |
| `--split-min-coverage-gain` | `0.05` | 拆分至少带来 5% 覆盖提升才接受 |
| `--part-gap-ratio` | `0` | 相邻部件允许共面接触，不人为添加缝隙 |
| `--no-refit-preserve-contact` | 不要使用 | 使用后会关闭源相邻标签的面接触保护 |
| `--no-adaptive-split` | 不要使用 | 使用后会关闭自动拆分 |

## 3A.3 AABB 输出

重点文件：

```text
03_fitobj_aabb/
├── mc_model_raw_aabb.glb
├── mc_model_before_surface.glb
├── mc_model_after_surface.glb
├── mc_model_after_surface.obj
├── paper_model.glb
├── paper_model.obj
├── paper_model.mtl
├── paper_model_texture.png
├── paper_model_parts.json
├── parts_raw_aabb.json
├── parts_before_surface.json
└── parts_after_surface.json
```

推荐：

- Blender 纹理预览：`paper_model.glb`
- Blender Paper Model 展开：`paper_model.obj`

---

# 3B. FitOBJ：Primitive Paper Model

Primitive 模式会根据每个 PartField cluster 的几何和目标面数，在以下候选中自动选择：

```text
box / prism / frustum / cone / ellipsoid / convex
```

V25 默认使用 `fixed` 接触模式：先重建并冻结原始 PartField 分割接口，再拟合接口之外的表面；复杂部件会自动使用局部非凸适配，不需要额外 CLI 参数。

## 3B.1 完整命令

```bash
rm -rf "$WORK/03_fitobj_primitive"

python -m partfield_mc.cli \
  "$SOURCE_GLB" \
  -o "$WORK/03_fitobj_primitive" \
  --postprocess-only \
  --normalized-mesh "$NORMALIZED_MESH" \
  --labels "$LABELS" \
  --clusters "$CLUSTERS" \
  --category animal \
  --up-axis y \
  --fit-mode primitive \
  --obj-mode surface \
  --primitive-types auto \
  --primitive-target-faces 0 \
  --primitive-max-faces 48 \
  --primitive-max-sides 24 \
  --primitive-fit-samples 2500 \
  --primitive-complexity-weight 0.025 \
  --primitive-contact-mode fixed \
  --primitive-contact-overlap-ratio 0 \
  --primitive-interface-max-sides 8 \
  --primitive-interface-min-width-ratio 0.006 \
  --primitive-interface-plane-tolerance-ratio 0.000001 \
  --primitive-connector-sides 4 \
  --primitive-connector-radius-ratio 0.028 \
  --primitive-connector-inset-ratio 0.28 \
  --primitive-connector-min-length-ratio 0.002 \
  --face-resolution 64 \
  --surface-samples 500000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap repeat \
  --padding 1
```

## 3B.2 Primitive 关键参数

| 参数 | 推荐值 | 作用 |
|---|---:|---|
| `--fit-mode primitive` | 必需 | 启用低面数多类型几何拟合 |
| `--obj-mode surface` | 必需 | 输出闭合 Paper Model shells |
| `--primitive-types` | `auto` | 启用全部候选类型；也可写 `box,prism,cone` |
| `--primitive-target-faces` | `0` | 根据该 cluster 的源三角面数自动估算目标纸模面数 |
| `--primitive-max-faces` | `48` | 单个部件允许的最大规范纸模面数 |
| `--primitive-max-sides` | `24` | prism/frustum/cone/ellipsoid 的最大环边数 |
| `--primitive-fit-samples` | `2500` | 每个 cluster 与候选模型的拟合评分采样数 |
| `--primitive-complexity-weight` | `0.025` | 面数复杂度惩罚；越高越偏向简单纸模 |
| `--primitive-contact-mode` | `fixed` | 冻结原始分割接口，只拟合其他表面 |
| `--primitive-contact-overlap-ratio` | `0` | 接口精确接触，不做隐藏插入 |
| `--primitive-interface-max-sides` | `8` | 冻结接口多边形的最大边数 |
| `--primitive-interface-min-width-ratio` | `0.006` | 原始界面退化时的备用最小半宽 |
| `--primitive-interface-plane-tolerance-ratio` | `0.000001` | 检查共享接口平面的数值容差 |
| `--no-primitive-resolve-overlaps` | 不要使用 | 默认会处理非邻接 shell 的意外重叠 |
| `--no-primitive-preserve-contacts` | 不要使用 | 使用后会关闭冻结接口/接触保持 |

下面四个 connector 参数主要用于源模型本身存在不连通组件时的兜底连接；正常 `fixed` 接口不会依靠它们连接相邻 PartField 部件：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--primitive-connector-sides` | `4` | 兜底 connector 的边数 |
| `--primitive-connector-radius-ratio` | `0.028` | connector 半径相对模型最长边的比例 |
| `--primitive-connector-inset-ratio` | `0.28` | 将接口中心向候选面内部移动，避免边/点接触 |
| `--primitive-connector-min-length-ratio` | `0.002` | connector 的最小长度比例 |

## 3B.3 Primitive 输出

重点文件：

```text
03_fitobj_primitive/
├── mc_model.glb
├── mc_model.obj
├── mc_model.mtl
├── mc_texture.png
├── parts.json
├── paper_model.glb
├── paper_model.obj
├── paper_model.mtl
├── paper_model_texture.png
└── paper_model_parts.json
```

推荐：

- Blender 纹理预览：`paper_model.glb`
- Blender Paper Model 展开：`paper_model.obj`
- 检查每个部件的类型、面数和接口：`paper_model_parts.json`

复杂多接口部件可能在 JSON 中显示：

```json
{
  "fitting_strategy": "local_patch_fit_around_fixed_source_interfaces",
  "fixed_interface_solver": "nonconvex_local_adapter",
  "source_interface_geometry_changed": false,
  "main_part_rigid_transform_applied": false
}
```

---

# AABB 与 Primitive 的选择

| 对比项 | AABB | Primitive |
|---|---|---|
| 基本形状 | 世界坐标轴对齐长方体 | 自动选择 box/prism/frustum/cone/ellipsoid/convex |
| 稳定性 | 更高 | 结构更复杂，对分割界面质量更敏感 |
| 外形相似度 | 较低，方块化明显 | 通常更接近源模型 |
| 面数 | 固定每块 6 个四边面 | 根据部件自动变化，最大由 `--primitive-max-faces` 控制 |
| 部件连接 | constrained AABB face contact | frozen PartField interface |
| 适合场景 | MC 风格、容易制作的纸模 | 更接近动物/人物外形的纸模 |
| 推荐导出 | `03_fitobj_aabb/paper_model.obj` | `03_fitobj_primitive/paper_model.obj` |

---

# 只重复运行 FitOBJ

Stage 1 和 Stage 2 成功后，后续修改拟合代码或参数只需：

```bash
cd /mnt/e/yp/partfield_hunyuan_v30
python -m pip install -e .

export WORK=$PWD/fox_stages
export CLUSTERS=8
source "$WORK/02_partfield/partfield_artifacts.env"
```

然后重新执行 **3A AABB** 或 **3B Primitive** 命令即可。

`--postprocess-only` 会跳过：

```text
Hunyuan3D
PartField feature inference
PartField clustering
```

---

# 常见问题

## 1. `--postprocess-only requires --normalized-mesh and --labels`

先加载 Stage 2 的环境变量文件：

```bash
source "$WORK/02_partfield/partfield_artifacts.env"
```

## 2. `labels` 数量和 `--clusters` 不一致

例如标签文件结尾是：

```text
_08.npy
```

则使用：

```bash
--clusters 8
```

## 3. 修改 Primitive 后是否需要重新跑 PartField？

不需要。只要 `SOURCE_GLB`、`NORMALIZED_MESH` 和 `LABELS` 没有变化，直接重跑 Stage 3。

## 4. 修改 cluster 数量后是否需要重新跑 PartField？

需要重新执行 PartField clustering，得到对应数量的 labels 文件。例如从 8 改为 10 后，需要 `_10.npy`，不能继续使用 `_08.npy`。

## 5. 最终应该导入哪个文件？

```text
查看纹理：paper_model.glb
Paper Model 展开：paper_model.obj
```


---

# V27 Hybrid Primitive 补充

苹果等连续主体推荐在 FitOBJ 阶段增加：

```bash
--fit-mode primitive \
--primitive-part-mode auto \
--primitive-contact-mode fixed
```

狐狸需要完全保持 V26 的“一条 PartField label 对应一个独立闭合壳体”时使用：

```bash
--fit-mode primitive \
--primitive-part-mode closed \
--primitive-contact-mode fixed
```

`auto` 不会修改 Hunyuan3D 或 PartField 的结果。它只在 FitOBJ 开始前，根据公共边界面积、边界长度、两侧面积比例以及部件细长度，决定哪些 PartField labels 是同一主体的表面分区，并在拟合前将它们合并。

---

# V28：Stage 3 混合拟合参数

V28 的 Hunyuan3D 和 PartField 阶段不变。只需升级代码后重新执行 Stage 3 FitOBJ。

## 苹果或连续主体

```bash
python -m partfield_mc.cli \
  "$SOURCE_GLB" \
  -o apple_primitive_v28 \
  --postprocess-only \
  --normalized-mesh "$NORMALIZED_MESH" \
  --labels "$LABELS" \
  --clusters 3 \
  --category generic \
  --up-axis y \
  --forward-axis auto \
  --fit-mode primitive \
  --primitive-part-mode auto \
  --obj-mode surface \
  --primitive-types ellipsoid,convex,frustum,cone,prism \
  --primitive-target-faces 0 \
  --primitive-max-faces 72 \
  --primitive-max-sides 24 \
  --primitive-fit-samples 3500 \
  --primitive-complexity-weight 0.012 \
  --primitive-contact-mode fixed \
  --primitive-interface-max-sides 12 \
  --primitive-interface-min-width-ratio 0.004 \
  --primitive-interface-plane-tolerance-ratio 0.000001 \
  --primitive-surface-main-body-min-area-ratio 0.35 \
  --primitive-surface-boundary-rings 0 \
  --primitive-surface-search-steps 18 \
  --face-resolution 96 \
  --surface-samples 500000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 2
```

## 狐狸完全保持旧 Closed Primitive

把以下参数改为：

```bash
--primitive-part-mode closed
```

其余 Primitive 参数可以继续使用。该模式不会启用主体受约束减面。


## V29 FitOBJ 说明

V29 修复 `Constrained surface boundary is not a collection of simple loops`。已有 Hunyuan3D 和 PartField 结果无需重跑，升级并执行 Stage 3 `--postprocess-only` 即可。苹果脚本：

```bash
SOURCE_GLB="$SOURCE_GLB" \
NORMALIZED_MESH="$NORMALIZED_MESH" \
LABELS="$LABELS" \
CLUSTERS=3 \
OUTPUT_DIR=apple_primitive_fitonly_v30 \
bash run_apple_primitive_v30_fitonly.sh
```


## V30 FitOBJ：固定接口局部侧向验证

V30 修复 constrained-surface 主体跨越接口无限平面时的误判。固定接口仍严格检查平面、顶点集合和面积，但部件侧向改为检查接口帽附近的相邻三角形，不再用整个主体所有顶点的全局中位数作为连通性门槛。苹果主体即使弯曲跨过接口平面，也不会因此触发 `Frozen source-interface validation failed`。

V30 还会在当前环境可用时优先调用 pymeshlab 的边界保持 QEM，将 dense constrained surface 降到纸模面数预算；接口边界坐标发生变化的结果会被拒绝并回退。

## V32 FitOBJ 自动接触模式补充

在 FitOBJ 阶段加入以下参数即可按接触强度自动处理：

```bash
--fit-mode primitive \
--primitive-contact-mode auto \
--primitive-contact-weak-threshold 0.20 \
--primitive-contact-strong-threshold 0.55 \
--primitive-contact-min-edge-count 6 \
--primitive-contact-medium-mode connector \
--primitive-surface-hard-max-faces 512 \
--primitive-validation-policy repair
```

弱连接允许不相交；中等连接默认使用 connector；强连接才冻结原始接口。`--postprocess-only` 的使用方式不变。
