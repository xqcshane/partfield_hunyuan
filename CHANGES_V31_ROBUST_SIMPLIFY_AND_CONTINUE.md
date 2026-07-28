# V31：强制减面与不中断鲁棒处理

V31 针对两类问题：

1. constrained surface 在接口约束较复杂时可能退回数万面，虽然流程成功，但没有达到纸模减面的目的；
2. 固定接口的辅助侧向判断失败时，流程直接中断，即使两侧接口顶点、平面和面积已经完全一致。

## 1. 减面成为必须执行的阶段

新增参数：

```bash
--primitive-surface-min-reduction-ratio 0.15
```

含义：constrained surface 至少应比输入外表面减少 15% 的三角面。

实际选择顺序：

1. 多组 PyMeshLab QEM 参数重试；
2. 保留边界拓扑，并将 QEM 轻微移动的边界顶点重新吸附到原始冻结接口坐标；
3. QEM 不可用或结果不安全时，搜索多组 boundary-locked source vertex clustering 分辨率；
4. 优先选满足最小减面比例且最接近目标面数的安全结果；
5. 如果达不到要求，但存在安全的减面结果，则使用面数最少的结果并记录 `minimum_reduction_relaxed=true`；
6. 只有所有安全减面器都失败时才保留源表面，并明确记录 `mandatory_reduction_failed=true`，不会静默假装减面成功。

输出元数据包括：

```json
{
  "source_outer_face_count": 46741,
  "simplified_outer_face_count": 430,
  "required_minimum_reduction_ratio": 0.15,
  "achieved_reduction_ratio": 0.9908,
  "mandatory_reduction_satisfied": true,
  "mandatory_reduction_failed": false,
  "simplifier": "pymeshlab_boundary_preserving_qem"
}
```

## 2. 固定接口采用分级验证，而不是一处异常就终止

新增参数：

```bash
--primitive-validation-policy repair
```

可选值：

- `repair`：默认。自动修复、继续并记录警告；
- `warn`：不自动修复，也不因几何验证失败终止；
- `strict`：保留以前的 fail-fast 行为。

### repair 模式

按以下顺序处理：

1. 恢复缺失的接口面索引；
2. 将两侧接口帽按循环顺序吸附到同一个 canonical frozen polygon；
3. 检查共享顶点、平面和面积；
4. 如果接口几何完全一致，但局部/全局侧向判断存在歧义，则接受该接触，并记录：

```json
{
  "connected": true,
  "strict_validation_passed": false,
  "accepted_with_warning": true,
  "connection_quality": "exact_geometry_side_ambiguous"
}
```

5. 如果连接树上的接口仍无法修复，则自动增加一个小型闭合 paper connector，保证整个装配体继续连通；
6. connector 也失败时，保留现有最佳结果并在 JSON 中记录失败边，不中断其他部件导出。

## 3. 单个部件拟合失败时的回退链

在 `repair` 或 `warn` 模式下，一个部件的主要拟合失败不会立即停止整个模型：

```text
constrained surface + fixed interface
→ constrained surface without fixed interface
→ closed primitive + fixed interface
→ closed primitive without fixed interface
→ contact stage repair / fallback connector
```

只有所有回退路径都无法生成任何闭合壳体时，才会报告终止性错误；这种情况通常表示输入损坏或代码缺陷，而不是普通几何歧义。

## 4. 苹果推荐参数

```bash
--fit-mode primitive \
--primitive-part-mode auto \
--primitive-contact-mode fixed \
--primitive-surface-min-reduction-ratio 0.15 \
--primitive-validation-policy repair
```

## 5. 狐狸兼容

狐狸继续使用：

```bash
--primitive-part-mode closed
```

同样可以加：

```bash
--primitive-validation-policy repair
```

这不会把狐狸的头、腿、尾巴改成 surface patch，只改变异常处理策略。
