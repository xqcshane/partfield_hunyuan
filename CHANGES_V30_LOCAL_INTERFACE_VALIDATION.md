# V30：固定接口局部侧向验证

## 修复的问题

V29 在固定接口验证阶段使用整个部件的非接口顶点中位数判断接口两侧。对于受约束苹果主体，主体可能跨越接口平面的无限延伸范围，导致接口几何完全正确但仍出现：

```text
normal=1 bulk=(positive, positive) opposite_sides=False
```

## V30 修改

1. 新增接口帽局部邻域 signed-distance 检查。
2. 只使用直接共享接口帽边的相邻面判断连接处材料位于哪一侧。
3. 全局 bulk 距离改为诊断信息，不再作为连通性硬门槛。
4. 局部环数值退化时，使用闭合壳体已定向接口帽的内侧方向回退。
5. 保持接口平面、顶点集合、面积及闭合拓扑检查不变。
6. 可选使用 pymeshlab 边界保持 QEM，优先将 constrained surface 降到纸模预算；只有接口边界坐标完全保留的结果才会被接受。

## 兼容性

- 苹果：`--primitive-part-mode auto` 使用 constrained surface。
- 狐狸：`--primitive-part-mode closed` 继续使用原闭合 primitive。
- Hunyuan3D 和 PartField 输出可继续复用，只需重新运行 `--postprocess-only`。

## 测试

通过 37 项测试，新增“全局 bulk 位于同侧、接口局部邻域位于正确两侧”的回归测试。
