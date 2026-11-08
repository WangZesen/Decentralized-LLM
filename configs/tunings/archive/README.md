# Historical tuning specifications

These 112 specifications were archived when the compact 40/60-TPP data-reuse
study became the current study. They remain tracked and usable through the
ordinary tuning CLI. Their filenames, numerical recipes, grids, seeds, and
controller settings are preserved; only the relative `base.configs` paths
changed to account for this subdirectory.

The archive includes:

- AdamW batch-64 joint LR/decay/beta searches and warmup variants.
- AdamW batch-128 20/40-TPP searches and beta2 extensions.
- The 40-TPP/reuse-2 full search and longer fixed-recipe comparisons.
- Fresh-data targeted runs through 120 TPP and repeated-order targeted runs.
- Batch-256 comparisons and Muon searches with momentum refinements.

Earlier campaign directories under `runs/` retain their frozen specifications,
source snapshots, results, and audit manifests. Archived public specifications
define fresh studies; use a campaign's saved state to resume its existing runs.

The 16 compact-study specifications remain in the [parent directory](../README.md).
