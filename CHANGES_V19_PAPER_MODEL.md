# V19: Canonical Blender Paper Model export

When `--obj-mode surface` or `--obj-mode all` is used, the pipeline now adds a
fourth export stage:

```text
paper_model.obj
paper_model.mtl
paper_model_texture.png
paper_model.glb
paper_model_parts.json
```

`paper_model.obj` is the canonical Blender input:

- one OBJ object named `paper_model`;
- one disconnected, independently closed shell per final cuboid;
- exactly 8 shared position vertices, 12 manifold edges and 6 quad faces per shell;
- no Boolean union;
- no vertex welding between touching cuboids;
- UV seams use OBJ's independent texture-coordinate indices and therefore do
  not split the geometry topology;
- exact contact rectangles are painted white in `paper_model_texture.png`, so
  contact regions remain untextured without subdividing the six cuboid faces.

`paper_model.glb` is a geometry-only one-node preview with the same closed shell
topology. Use `paper_model.obj` rather than the GLB for Blender's Export Paper
Model workflow because OBJ preserves shared geometry vertices and independent
UV indices simultaneously.
