# V29：Constrained Surface 边界拓扑修复

V28 在合并多个 PartField 主体表面分区后，假设剩余边界一定是若干互不接触的简单闭环。实际模型中，三个 label 可能在同一个源顶点相遇，或者同一接口的多个边界环在一个顶点处接触，从而形成边界度数 4 的 pinched vertex。V28 因此会报错：

```text
Constrained surface boundary is not a collection of simple loops
```

V29 的修改：

- 在 constrained surface 拟合前分析每个顶点周围的 incident-face fan；
- 当多个互不连通的 face fan 只共享同一坐标顶点时，复制该拓扑顶点；
- 复制只改变拓扑索引，不移动任何几何坐标；
- 将接触于一点的多个边界环拆成独立简单闭环；
- 在固定接口折叠后和 vertex clustering 后再次执行拓扑修复；
- topology-safe 检查现在同时验证边界顶点度数；
- `paper_model_parts.json` 增加 `source_patch_topology_repair`，记录拆分的 pinched vertex 数量；
- `--primitive-part-mode closed` 路径未修改，狐狸旧的独立闭合 primitive 行为保持不变。

V29 通过 36 项测试，包括两个边界盘仅共享一个源顶点的回归测试。
