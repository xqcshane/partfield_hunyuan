# V23: Stationary primitive parts with explicit paper connectors

V22 restored connectivity by rotating and translating the complete child
primitive until two existing faces became coplanar. On dense multi-part animal
models this could cascade through the contact tree, move limbs or ears away from
their fitted locations, and accept narrow edge-like contact regions.

V23 changes primitive contact fitting as follows:

- Source primitive vertices are never changed in the default `connector` mode.
- Parent and child faces are selected jointly using seam proximity, opposing
  normals, connector direction, face area, and usable interior patch radius.
- The interface centre is inset from the face boundary to prevent point/edge
  joints.
- Existing positive-area coplanar contacts are reused directly.
- Otherwise a closed low-face frustum connector is inserted. Both end polygons
  lie inside the selected parent/child faces and provide a known positive
  contact area.
- Connector pieces are independent watertight shells suitable for Blender Paper
  Model unfolding and glue-tab generation.
- The V22 rigid relocation strategy remains available with
  `--primitive-contact-mode move` for compatibility.

New CLI parameters:

```text
--primitive-contact-mode connector
--primitive-connector-sides 4
--primitive-connector-radius-ratio 0.028
--primitive-connector-inset-ratio 0.28
--primitive-connector-min-length-ratio 0.002
```
