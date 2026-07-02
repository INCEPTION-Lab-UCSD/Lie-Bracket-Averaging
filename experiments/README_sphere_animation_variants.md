# Sphere Animation Alternatives

This branch contains experimental renderings for the global ES sphere example.
The driver is:

```bash
.venv/bin/python experiments/sphere_animation_alternatives.py --preview
```

MP4 animations and matching PNG snapshots are written to:

```text
figures/sphere_animation_variants/
```

## Variants

- `paper_markers`: grayscale `J(x)` sphere with larger high-contrast start and target markers.
- `high_contrast`: cividis cost surface with brighter trajectory and marker treatment.
- `cost_dashboard`: sphere animation plus a cost-over-time panel with an initial-cost reference line.
- `jump_dashboard`: sphere animation plus transformed costs, jump margin, mode trace, and actual switch markers.

Render one variant:

```bash
.venv/bin/python experiments/sphere_animation_alternatives.py --variant cost_dashboard --preview
```

Render full-resolution MP4/PNG pairs by omitting `--preview`.

```bash
.venv/bin/python experiments/sphere_animation_alternatives.py --variant all
```
