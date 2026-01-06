# data layout

local data is the source of truth. use this layout and publish versioned snapshots to hugging face for training runs.

- `raw/`: source archives or downloaded data.
- `processed/`: cleaned and normalized outputs.
- `stages/stage1/`: stage 1 mixtures.
- `stages/stage2/`: stage 2 mixtures.
- `stages/stage3/`: stage 3 mixtures.
- `arc/`: arc-agi data, synthetic grids, and transforms.
- `metadata/`: manifests, hashes, filters, and provenance logs.

## candidate sources (short list)
- clean web/edu: fineweb-edu, openwebtext2, wikipedia, pg-19/public domain books.
- code: the stack v2 (permissive subset), starcoder2data-edu.
- math: gsm8k, mathqa, openwebmath.
- arc-like: arc-agi (public), arc-agi-2 public.
- puzzle/logic: sudoku-extreme, maze-hard, synthetic grid transforms aligned with arc-agi.
