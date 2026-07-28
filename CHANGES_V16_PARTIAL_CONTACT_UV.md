# v16: drop enclosed parts and untexture exact contact patches

1. Large-first overlap resolution never relocates a fully enclosed smaller cuboid. If no positive outside slab remains, the smaller cuboid is removed from the after-surface result.
2. Post-surface faces are partitioned on the exact cuboid-contact rectangle grid. Visible sub-regions keep their source UVs; contact/intersection sub-regions remain geometry but use an untextured material.
3. OBJ contact patches use `mc_hidden` and omit `/vt` indices. GLB writes textured and contact-untextured patch meshes separately.
4. `parts_after_surface.json` records requested, kept and dropped segment IDs. Before-surface exports remain unchanged.
