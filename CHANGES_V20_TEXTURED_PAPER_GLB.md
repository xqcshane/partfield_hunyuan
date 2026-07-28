# V20: Textured Paper Model GLB

`paper_model.glb` now embeds the paper texture atlas and UV coordinates.

- `paper_model.glb`: one Blender object with embedded texture, intended for easy visual inspection.
- `paper_model.obj` + `paper_model.mtl` + `paper_model_texture.png`: canonical paper-unfolding input, preserving 8 shared geometry vertices and 6 quads per cuboid shell.

Because glTF stores one UV per position vertex, the textured GLB uses 24 render vertices per cuboid at UV seams. This is normal and does not change the cuboid shape. Use the OBJ for topology-sensitive paper unfolding.
