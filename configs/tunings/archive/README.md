# Historical tuning specifications

These 143 specifications include the 112 previously archived studies and all 31
specifications that preceded the current 20M, 4-TPP-warmup tuning round. They
remain tracked and usable through the ordinary tuning CLI. Their filenames,
numerical recipes, grids, seeds, and controller settings are preserved; only the
relative `base.configs` paths changed when files moved into this subdirectory.

The archive includes:

- AdamW batch-64 joint LR/decay/beta searches and warmup variants.
- AdamW batch-128 20/40-TPP searches and beta2 extensions.
- The 40-TPP/reuse-2 full search and longer fixed-recipe comparisons.
- Fresh-data targeted runs through 120 TPP and repeated-order targeted runs.
- Batch-256 comparisons and Muon searches with momentum refinements.
- The 16 compact 40/60-TPP data-reuse searches with 3-TPP warmup.
- The eight 50M study specifications, earlier platform presets, and tuning examples.

The completed compact data-reuse campaign is
`runs/p-tuning/20m-512c-128bs-40repeat-60tpp-local-tuning-warmup3-20261107`.
Earlier campaign directories under `runs/` retain their frozen specifications,
source snapshots, results, and audit manifests. Archived public specifications
define fresh studies; use a campaign's saved state to resume its existing runs.

The 21 current specifications cover seven method/worker combinations at 20, 40,
and 60 TPP and are indexed in the [parent directory](../README.md).
