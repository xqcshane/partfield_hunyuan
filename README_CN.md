# NVIDIA PartField → MC 方块人物/动物完整流水线

本项目把已有 `GLB/OBJ` 模型处理为：

```text
输入模型
  ↓
NVIDIA PartField 部件分割
  ↓
每个部件拟合 OBB/AABB 长方体
  ↓
从原始 UV 贴图烘焙每个长方体的六个面
  ↓
mc_model.glb + mc_model.obj + mc_texture.png + parts.json
```

## 重要说明

1. PartField 是**类别无关的层级部件分割模型**。它输出 `segment_id`，不会天然输出“头、腿、尾巴”等文本语义。
2. `--category animal/person` 只做基础几何命名，名称是启发式结果；`segment_id` 和 3D 位置、尺寸、旋转才是可靠输出。
3. PartField 官方许可证仅允许非商业研究和教育用途。商业产品不能直接使用该模型或衍生代码。
4. 本压缩包不重新分发 NVIDIA PartField 源码或权重；安装脚本会从官方仓库和官方 Hugging Face 权重地址下载。

## 1. 环境

推荐：

- Ubuntu / WSL2
- NVIDIA GPU
- Python 3.10
- PyTorch 2.4
- CUDA 12.4

官方环境较重，建议新建独立 Conda 环境，不要安装到 `palettenerf`。

## 2. 一键安装 PartField

```bash
cd partfield_mc_pipeline
bash setup_partfield_wsl.sh /mnt/e/yp/PartField
```

然后：

```bash
conda activate partfield
```

如已安装官方 PartField，只需要安装本项目的包装层依赖：

```bash
python -m pip install -r requirements_wrapper.txt
python -m pip install -e .
```

## 3. 完整运行 mouse.glb

```bash
partfield-mc "glb/mouse.glb" \
  -o mouse_partfield_mc \
  --partfield-repo /mnt/e/yp/PartField \
  --checkpoint /mnt/e/yp/PartField/model/model_objaverse.ckpt \
  --clusters 9 \
  --category animal \
  --fit-mode obb \
  --face-resolution 64 \
  --surface-samples 500000 \
  --palette-size 0
```

输出：

```text
mouse_partfield_mc/
├── mc_model.glb
├── mc_model.obj
├── mc_model.mtl
├── mc_texture.png
├── parts.json
├── partfield_segmented.ply
├── partfield_clusters/
└── partfield_work/
```

## 4. 参数建议

### 控制主体方块数量

```bash
--clusters 6
--clusters 9
--clusters 12
```

`6~10` 更像 MC 大块主体；`15~20` 会出现更多小部件。

### OBB 和 AABB

```bash
--fit-mode obb
```

保留腿、尾巴等部件的旋转方向。

```bash
--fit-mode aabb
```

所有部件都与世界坐标轴对齐，方块感更强，但可能包围较多空白。

### MC 网格吸附

```bash
--grid-divisions 32
```

将位置和尺寸吸附到整个模型最长边的 `1/32` 网格。默认 `0` 表示不吸附。

### 贴图细节

```bash
--face-resolution 64 --surface-samples 500000
```

更高清：

```bash
--face-resolution 128 --surface-samples 1000000
```

低内存：

```bash
--face-resolution 32 --surface-samples 150000
```

### PartField 显存不足

```bash
--n-point-per-face 200 --n-sample-each 3000
```

官方说明显存不足时可以降低 `n_point_per_face`。

### 网格连接很乱

默认使用：

```bash
--clustering agglo --adjacency mst
```

若结果不理想，可以测试：

```bash
--clustering kmeans
```

或者：

```bash
--preprocess-mesh
```

注意：预处理会改变 PartField 内部网格的面结构，但本程序仍使用其对应输出进行长方体拟合。

## 5. 只运行后处理

PartField 已经运行完时，可以直接传入：

- `input_UID_0.ply`
- `UID_0_NN.npy`

```bash
python run_partfield_mc.py "glb/mouse.glb" \
  -o mouse_postprocess \
  --postprocess-only \
  --normalized-mesh /path/to/input_mouse_0.ply \
  --labels /path/to/mouse_0_09.npy \
  --clusters 9 \
  --category animal \
  --face-resolution 64
```

## 6. `parts.json`

每个部件包含：

```json
{
  "name": "part_03",
  "segment_id": 3,
  "center": [0.1, 0.3, -0.2],
  "size": [0.4, 0.2, 0.7],
  "rotation": [[1,0,0],[0,1,0],[0,0,1]],
  "transform": [[1,0,0,0.1],[0,1,0,0.3],[0,0,1,-0.2],[0,0,0,1]]
}
```

这就是自动生成 `build_xxxxx_template()` 所需要的 3D 部件位置、尺寸和旋转，不需要再把模板写死在 Python 中。

## 7. 测试后处理代码

不需要 PartField 权重即可测试长方体拟合和贴图导出：

```bash
pytest -q
```

## OBJ 导出模式

参数：

```bash
--obj-mode merged|separate|surface|all
```

- `merged`：一个 `mc_model.obj`，保留所有长方体的全部面，允许方块相互穿插。
- `separate`：在 `obj_parts/` 中为每个 PartField cluster 输出一个独立 OBJ。
- `surface`：默认使用非重叠约束重新拟合。每个小部件根据自己的 PartField 源面重新搜索 AABB，而不是对已经生成的方块做后裁剪；源网格中相邻的标签会优先保持共面连接。接触子区域保留几何但不贴图。
- `all`：同时输出 merged、严格 surface 和逐部件 OBJ。

## 严格 cluster surface 模式

使用：

```bash
python -m partfield_mc.cli input.glb \
  -o output \
  --clusters 8 \
  --fit-mode aabb \
  --obj-mode surface
```

默认几何策略是：

```text
raw AABB
  ↓
non-overlap constrained refit
  ↓
surface material processing
```

约束规则：

1. 每个 PartField cluster 先生成一个原始 AABB，保存到 `mc_model_raw_aabb.*`，该诊断模型允许重叠。
2. 随后按源 cluster 的真实表面积排序，优先固定主体，而不是按可能被弯曲尾巴放大的 AABB 体积排序。
3. 对每个后续 cluster，从它自己的源三角面重新搜索满足约束的 AABB。候选方块必须与已经固定的方块没有正体积重叠。
4. 搜索目标优先保留源表面积覆盖率、源中心和有效体积。
5. 如果两个 PartField 标签在原网格中共享边，搜索会优先令新的 AABB 与对应主体面共面接触，避免尾巴、腿和头部在去重叠后断开。
6. 完全位于主体内部且找不到有效外侧候选的小 cluster 会直接删除，不会移动到模型外侧。
7. `before_surface` 已经是最终非重叠几何；`after_surface` 不再裁剪或移动方块，只处理接触区域的材质。
8. 接触矩形保留闭合几何，但不分配图片纹理；其余可见区域继续使用 UV。

可调参数：

```bash
--surface-fit-strategy refit
--refit-min-coverage 0.05
--refit-beam-width 64
```

关闭相邻部件连接优先级：

```bash
--no-refit-preserve-contact
```

退回旧版后裁剪方式：

```bash
--surface-fit-strategy trim
```

如果设置正数间隔：

```bash
--part-gap-ratio 0.001
```

方块之间会主动留缝，因此不能同时要求共面连接。通常建议保持默认 `0`。

对于 8-cluster 命令：

```bash
python -m partfield_mc.cli \
  fox_result/hunyuan3d_textured_multiview.glb \
  --partfield-repo /mnt/e/yp/PartField \
  --checkpoint /mnt/e/yp/PartField/model/model_objaverse.ckpt \
  --simplify-faces 5000 \
  --n-point-per-face 50 \
  --n-sample-each 1000 \
  -o fox_mc_result \
  --clusters 8 \
  --fit-mode aabb \
  --obj-mode surface
```

`parts_before_surface.json` 中每个部件会记录：

- `raw_aabb_center`、`raw_aabb_size` 和 `raw_aabb_volume`
- `constrained_refit_priority_rank`
- `constrained_refit_coverage_ratio`
- `constrained_refit_contact_segments`
- `constrained_refit_volume_ratio`
- `nonoverlap_constraint_satisfied`

## v8 Hunyuan3D Integration

v8 新增图片生成三维模型入口：

```bash
partfield-mc \
 --image mouse.png \
 --hunyuan-repo /path/to/Hunyuan3D \
 -o mouse_result
```

流程：

Image -> Hunyuan3D -> GLB -> PartField -> MC Cuboid -> OBJ/GLB

原有模型处理方式保持兼容：

```bash
partfield-mc model.glb -o output
```

## v17：原始 AABB、约束拟合和 Surface 三阶段输出

使用 `--obj-mode surface` 或 `--obj-mode all` 时，会保存：

```text
partfield_input_simplified.glb
    简化后的三角网格，也是实际传给 PartField 的输入。

mc_model_raw_aabb.glb/.obj
parts_raw_aabb.json
    每个 cluster 直接拟合的原始 AABB，允许重叠，用于诊断。

mc_model_before_surface.glb/.obj
parts_before_surface.json
    非重叠约束重新拟合后的闭合长方体。该阶段几何已经最终确定。

mc_model_after_surface.glb/.obj
parts_after_surface.json
    与 before_surface 使用完全相同的中心和尺寸；只把共面接触矩形改成无图片纹理材质。
```

兼容输出 `mc_model.glb/.obj` 仍然等同于 `after_surface`。

CLI JSON 还会返回：

```text
raw_aabb_glb
raw_aabb_obj
raw_aabb_parts_json
before_surface_glb
before_surface_obj
before_surface_parts_json
after_surface_glb
after_surface_obj
after_surface_parts_json
```

## V18：脸部优先与躯干自适应拆分

V18 不使用 LLM Agent，而是使用确定性的几何分类与约束优化：

```text
PartField clusters
  ↓
自动推断 body / face / tail / limb
  ↓
先锁定脸部 AABB
  ↓
躯干在不侵入脸部的条件下重新拟合
  ↓
单个躯干盒覆盖不足时，自动拆成两个相接且不重叠的 AABB
  ↓
其他部件继续拟合
```

推荐命令：

```bash
python -m partfield_mc.cli \
  fox_result/hunyuan3d_textured_multiview.glb \
  --partfield-repo /mnt/e/yp/PartField \
  --checkpoint /mnt/e/yp/PartField/model/model_objaverse.ckpt \
  --simplify-faces 5000 \
  --n-point-per-face 50 \
  --n-sample-each 1000 \
  -o fox_mc_result_v18 \
  --clusters 8 \
  --fit-mode aabb \
  --obj-mode surface \
  --semantic-refit animal \
  --max-extra-cuboids 1 \
  --protected-min-coverage 0.85 \
  --split-min-coverage-gain 0.05
```

参数含义：

- `--clusters 8`：PartField 仍生成 8 个源 cluster。
- `--semantic-refit animal`：使用动物的脸部优先几何规则；不是 Agent。
- `--max-extra-cuboids 1`：允许自适应拆分后最多比 cluster 数多 2 个方块。
- `--protected-min-coverage 0.85`：保护脸部后，单个躯干方块覆盖不足 85% 时考虑拆分。
- `--split-min-coverage-gain 0.05`：拆分至少提升 5% 源表面积覆盖率才采用。

如自动分类错误，可先测试：

```bash
--semantic-refit off --no-adaptive-split
```

这会退回 V17 的纯表面积优先重新拟合。

## V21：按部件自动选择低面数原型并生成 Paper Model

新增拟合方式：

```bash
--fit-mode primitive
```

它不再强制把每个 PartField cluster 都变成长方体，而是在 **PartField 已使用的减面网格** 上，对每个 cluster 独立执行：

```text
简化后的 cluster 三角面数
  ↓ 自动计算该部件的 paper-face 目标数量
生成多种候选：
  box / prism / frustum / cone / ellipsoid / convex polyhedron
  ↓
比较源表面→候选、候选→源表面的双向几何误差
  + 候选面数与目标面数的差异
  + Paper Model 装配复杂度
  ↓
选择最接近的模型类型和面数
  ↓
生成共享几何顶点、闭合的多面体 shell
  ↓
读取原始 PartField label 的公共边界，并先重建低面数共享接口多边形
  ↓
接口的位置、方向、面积和顶点坐标冻结不变
  ↓
所有 primitive 候选只在接口外侧拟合；候选越过接口平面的部分会被裁掉
  ↓
父子部件分别包含同一组接口顶点，形成精确的面—面接触
  ↓
paper_model.obj + 贴图 + GLB 预览
```

推荐命令：

```bash
python -m partfield_mc.cli \
  fox_result/hunyuan3d_textured_multiview.glb \
  --partfield-repo /mnt/e/yp/PartField \
  --checkpoint /mnt/e/yp/PartField/model/model_objaverse.ckpt \
  --simplify-faces 5000 \
  --n-point-per-face 50 \
  --n-sample-each 1000 \
  -o fox_primitive_paper \
  --clusters 8 \
  --fit-mode primitive \
  --obj-mode surface \
  --category animal \
  --primitive-target-faces 0 \
  --primitive-max-faces 48 \
  --primitive-max-sides 24 \
  --primitive-fit-samples 2500 \
  --primitive-contact-mode fixed \
  --primitive-interface-max-sides 8 \
  --primitive-interface-min-width-ratio 0.006 \
  --face-resolution 64 \
  --surface-samples 500000
```

主要参数：

- `--primitive-types auto`：比较全部候选。也可指定 `box,prism,frustum,cone,ellipsoid,convex` 的子集。
- `--primitive-target-faces 0`：根据每个 cluster 在减面网格中的三角面数，自动计算目标 paper face 数；指定正整数时，所有部件使用相同目标。
- `--primitive-max-faces 48`：单个部件允许的最大纸模面数。
- `--primitive-max-sides 24`：棱柱、台体、锥体和低面椭球的最大环向分段数。
- `--primitive-fit-samples 2500`：每个候选的几何评分采样数。增大可提高稳定性，但会增加运行时间。
- `--primitive-complexity-weight 0.025`：面数复杂度在总评分中的权重。增大后会更偏向简单纸模。
- `--no-primitive-resolve-overlaps`：关闭非相邻部件之间的意外重叠调整；原本相邻的 PartField 部件不会被该步骤推开。
- `--no-primitive-preserve-contacts`：关闭 primitive 部件连接恢复。纸模输出通常不要使用此参数。
- `--primitive-contact-mode fixed`：V24 默认模式。先冻结源分割接口，再对接口之外的外表面进行 primitive 拟合。
- `--primitive-interface-max-sides 8`：每个冻结接口允许的最大边数。数值越高越接近源边界，数值越低越容易制作。
- `--primitive-interface-min-width-ratio 0.006`：源边界退化为线或点时，备用接口矩形的最小半宽。正常闭合边界不会使用该值。
- `--primitive-interface-plane-tolerance-ratio 1e-6`：验证父子接口顶点和共面性的数值容差。
- `--primitive-contact-mode connector`：保留 V23 行为；主体不移动，但会在两个已有 primitive 之间增加连接件。
- `--primitive-contact-mode move`：保留 V22 行为；会旋转和平移子部件，一般不用于最终纸模。
- `--primitive-connector-*`：仅用于 `connector` 模式，或在 `fixed` 模式下连接源模型中原本不相连的独立组件。
- `--primitive-contact-overlap-ratio 0`：仅用于 `move` 旧模式。

输出：

```text
mc_model.obj/.glb
mc_texture.png
parts.json

paper_model.obj
paper_model.mtl
paper_model_texture.png
paper_model.glb
paper_model_parts.json
```

`paper_model.obj` 的特点：

- 一个 OBJ object；
- 每个 PartField cluster 仍是一个独立闭合 shell；
- 每个原始 PartField 公共边界只重建一次，并作为父子部件完全相同的冻结接口面；
- 主体 primitive 的外表面围绕冻结接口拟合，接口不会随候选类型改变；
- 父子接口顶点坐标完全一致、法线相反、接触面积相同；
- 主体 primitive 不会为了接触而发生整体旋转、平移或统一缩放；
- 只有源模型本身包含不连通组件时，才会增加一个闭合 connector shell 作为跨组件连接件；
- 相邻面共享几何顶点，UV 接缝只使用独立的 OBJ texture indices；
- 保留三角面、四边面和凸 n-gon 的真实 paper face 类型；
- 不做 Boolean union，也不跨部件焊接顶点；
- 可直接作为 Blender Paper Model 展开输入。

`parts.json` 和 `paper_model_parts.json` 会记录：

- `source_face_count`：减面后的源 cluster 三角面数；
- `target_face_count`：自动或手动指定的目标纸模面数；
- `primitive_type`：最终选择的原型类型；
- `paper_face_count` 和 `triangle_count`；
- `fit_score`；
- `top_candidates`：排名靠前的候选类型、面数和各项几何误差。
- `contact_tree_parent_segment_id`：该部件连接到哪个父部件；
- `contact_tree_parent_face_index` / `contact_tree_child_face_index`：实际用于纸模连接的两个面；
- `frozen_interface_face_indices`：该部件与各相邻部件共享的固定接口面索引；
- `frozen_interface_areas`：每个冻结接口的实际面积；
- `contact_tree_contact_mode`：V24 源相邻部件为 `fixed_interface`；跨不连通组件时为 `connector_patch`；
- `contact_tree_connector_segment_id`：仅跨组件连接件具有负数 segment id；
- `contact_tree_contact_area`：冻结接口或连接件端面的面积；
- `contact_tree_main_part_moved`：V24 应为 `false`；
- `source_interface_geometry_changed`：V24 应为 `false`；
- `contact_graph_connected`：全部保留部件是否已经连接成一个装配结构。

## V24：冻结初始接口后再拟合外表面

V23 仍然是在 primitive 拟合完成后处理连接，因此连接面由拟合结果决定。V24 将顺序反过来：

1. 先从原始 PartField face labels 的公共边界恢复接口位置、方向和边界点；
2. 将边界投影为一个低面数凸多边形，并在父子部件之间只创建一次；
3. 该多边形的三维顶点被同时写入两个相邻部件，且整个拟合过程中保持不变；
4. 每个 box、prism、frustum、cone、ellipsoid 或 convex 候选都先按接口半空间裁剪；
5. 再以冻结接口顶点和候选外部支撑点重建闭合凸壳，并重新计算几何评分；
6. 最终选择的是“满足固定接口约束后的最佳候选”，而不是先选 primitive 再补连接；
7. overlap resolver 不再移动含冻结接口的部件，只报告非相邻部件的剩余重叠，避免破坏接口。

推荐检查：

```text
primitive_contact_mode: fixed
fitting_strategy: fit_outer_surface_around_fixed_source_interfaces
contact_tree_contact_mode: fixed_interface
main_part_rigid_transform_applied: false
source_interface_geometry_changed: false
connector_count: 0  # 源模型连通时
```

## V22：Primitive 纸模部件连接约束

V21 的 primitive 候选是逐 cluster 独立拟合的，因此即使原始 PartField
分割在同一表面上连续，拟合后的头、躯干、四肢或尾巴也可能出现间隙。
V22 增加以下固定流程：

1. 从 `mesh.face_adjacency` 和 PartField face labels 恢复 label 邻接图及公共边界位置；
2. 以面积最大的部件为根，建立覆盖全部部件的接触树；
3. 相邻部件只处理非接触方向的意外碰撞，不再被 overlap resolver 推开；
4. 在父、子 primitive 上选择靠近源边界并朝向正确的现有多边形面；
5. 对子 primitive 做刚体旋转和平移，使两个面法向相反、平面重合并有正接触面积；
6. 若源模型本身含多个不连通组件，则使用最近的两个部件补充连接边，确保最终装配体整体连通。

默认命令不需要新增参数：

```bash
--fit-mode primitive --obj-mode surface
```

检查 `paper_model_parts.json` 中所有部件的：

```text
contact_graph_connected: true
```

即可确认连接约束执行成功。

## V23：固定主体 + 显式纸模连接件

V22 为了使两个现有 primitive 面共面，会旋转和平移整个子部件。复杂动物模型中，
这种级联刚体变换容易把耳朵、腿、尾巴或头部拉到错误位置，并且可能只形成很小的
边接触。V23 改为：

1. 任何 fitted source part 都保持 primitive 拟合结束时的位置和朝向；
2. 父子接触面成对评分，评分同时考虑源边界距离、法线朝向、两面是否相对、面内可用半径和连接长度；
3. 接触中心向面内部移动，禁止把尖角或边界点当作有效接触面；
4. 已有正面积共面接触时直接复用；
5. 否则生成一个闭合低面数连接棱台，连接件两端完整落在父子候选面内部；
6. 连接件作为独立纸模 shell 导出，可以单独展开并在两端加 glue tabs；
7. `paper_model_parts.json` 分别记录 `source_part_count` 和 `connector_count`。

推荐检查：

```text
primitive_contact_mode: connector
contact_tree_main_part_moved: false
contact_graph_connected: true
```


## V25：多接口局部约束拟合

V24 要求一个分割块的所有冻结接口都同时成为同一个凸多面体的外表面。躯干同时连接头、尾巴和多条腿时，这些接口平面通常不构成一个兼容的凸体，因此会出现：

```text
No primitive candidate can preserve the frozen interfaces
```

V25 保留原始接口的位置、顶点、面积和方向不变，但不再强制整个部件保持凸形：

1. 优先尝试 V24 的凸约束拟合；
2. 凸约束不成立时，自动切换为 `nonconvex_local_adapter`；
3. 为每个接口联合选择 primitive 上对应的局部三角面；
4. 只重建该局部区域，使其连接到冻结接口；
5. 不移动完整部件，不添加独立 connector；
6. 最终仍是一个闭合、共享顶点、可展开的纸模壳体。

无需增加新的 CLI 参数，继续使用：

```bash
--fit-mode primitive \
--primitive-contact-mode fixed
```

`paper_model_parts.json` 中可检查：

```text
fitting_strategy: local_patch_fit_around_fixed_source_interfaces
fixed_interface_solver: nonconvex_local_adapter
source_interface_geometry_changed: false
```
