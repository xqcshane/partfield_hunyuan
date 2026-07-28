# V22: contact-constrained primitive paper assembly

- Recovers PartField label adjacency and weighted boundary anchors from shared source-mesh edges.
- Builds a large-part-rooted contact tree covering every surviving segment.
- Protects source-adjacent pairs from the non-adjacent overlap-separation pass.
- Selects existing parent/child primitive faces near each source boundary.
- Applies a rigid child rotation and translation so the selected faces are coplanar, oppositely oriented, and share positive area.
- Preserves every primitive as an independently watertight shell suitable for unfolding.
- Adds fallback nearest-component joints when the source mesh itself is disconnected.
- Writes contact face indices, area, plane error, rigid transform, and global connectivity status into JSON metadata.
- Adds `--no-primitive-preserve-contacts` and `--primitive-contact-overlap-ratio`.
