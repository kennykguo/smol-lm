 # plan: smol llm (100m) with kda + trm variants

## goals and inputs
1. **confirm constraints and references**: treat `nanogpt.md`, `smol-llm.md`, `ultrascale-playbook.md`, `kda.md`, and `trm.md` as primary guidance for scope, staged training, eval-first, and architecture details. this keeps the plan aligned with the requested documents.
2. **target specs**: build two fully separate end-to-end pipelines:
   - **route a**: dense 100m model with a 3:1 kda-to-full-attention layer ratio (per `kda.md`).
   - **route b**: tiny recursive model (trm) 100m model using the same 3:1 kda ratio inside the recursive core (per `trm.md` + `kda.md`).
3. **framework**: implement both architectures in pytorch.

## environment setup (uv + notebooks)
3. **create a uv-managed environment**: use `uv` to set up python, dependencies, and jupyter. this follows the “do it right, don’t cut corners” approach and keeps reproducibility.
4. **notebook-first workflow**: create notebooks for data prep, sanity checks, and small-scale runs. use scripts only when a repeatable batch pipeline is required.
5. **concrete setup steps**:
   - `uv venv .venv`
   - `uv pip install -r requirements.txt`
   - `uv pip install ipykernel jupyter`
   - `IPYTHONDIR=./.ipython JUPYTER_DATA_DIR=./.jupyter ./.venv/bin/python -m ipykernel install --user --name smol-llm`
6. **notebook scaffolding**:
   - `notebooks/00_env_smoke.ipynb` (imports, cuda check, tiny forward pass)
   - `notebooks/10_data_prep.ipynb` (data layout + validation)
   - `notebooks/20_route_a_train.ipynb` (dense kda)
   - `notebooks/30_route_b_train.ipynb` (trm + kda)

## data strategy (arc-agi focus + staged training)
7. **define staged data curriculum** (per multi-stage training in `smol-llm.md`):
   - **stage 1 (broad generalization)**: web + code + math (light), focusing on clean reasoning-friendly text.
   - **stage 2 (reasoning boost / continued pretraining)**: mix in instruction and reasoning corpora; add structured puzzle data.
   - **stage 3 (arc-agi specialization)**: heavier arc-like data and program synthesis traces; smaller, high-quality sets late for maximum impact.
8. **canonical data flow (local + versioned hf snapshots)**:
   - **source of truth**: local file-structured dataset in this repo (or sibling storage).
   - **snapshots**: publish immutable, versioned hugging face datasets (e.g., `v0.1`, `v0.2`) for kaggle training.
   - **why**: local iteration is simplest; hf snapshots give reproducible training and easy transfer to kaggle.
9. **local data layout (file structure)**:
   - `data/raw/` original sources or downloaded archives.
   - `data/processed/` cleaned text, normalized formats, and tokenized shards.
   - `data/stages/stage1/`, `data/stages/stage2/`, `data/stages/stage3/` curated mixtures per stage.
   - `data/arc/` arc-agi datasets, synthetic grid tasks, and derived transforms.
   - `data/metadata/` manifests, filters, and provenance logs.
10. **snapshot workflow (hf)**:
   - freeze a stage folder into a manifest (file list + hashes).
   - upload as a versioned dataset: `smol-llm-stage1:v0.1`.
   - pin the exact snapshot tag in kaggle training runs.
11. **curation pipeline (repeatable steps)**:
   - ingest and normalize sources into `data/processed/`.
   - deduplicate and filter low-quality content; record filters in `data/metadata/`.
   - tokenize into shards per stage; emit `manifest.jsonl` with file hashes.
   - build stage mixtures by manifest composition, not by copying files.
12. **candidate data sources (initial short list)**:
   - clean web/edu: fineweb-edu, openwebtext2, wikipedia, pg-19/public domain books.
   - code: the stack v2 (permissive subset), starcoder2data-edu.
   - math: gsm8k, mathqa, openwebmath.
   - arc-like: arc-agi (public), arc-agi-2 public.
   - puzzle/logic: sudoku-extreme, maze-hard, synthetic grid transforms aligned with arc-agi.
13. **proposed mixture (initial)**:
   - 50% clean web/edu text, 20% code, 10% math, 10% synthetic program synthesis (grid/graph/puzzle), 10% arc-like tasks and derived transforms.
   - adjust per stage: shift to 30/20/10/20/20 by stage 3.
   - rationale: `smol-llm.md` emphasizes staged mixtures, late high-quality data, and eval-driven adjustments.

## architecture route a (dense kda hybrid)
14. **dense hybrid baseline**:
   - start from a standard gpt-like stack (use `nanogpt.md` for repo structure).
   - replace attention blocks with 3:1 kda-to-full-attention layers as in `kda.md`.
   - keep parameter budget ~100m by adjusting depth/width and tying embeddings if needed.
15. **why**: `kda.md` shows kda hybrid keeps quality with memory/throughput gains. this is a clean baseline to measure benefits.

## architecture route b (trm + kda)
16. **trm adaptation**:
   - implement trm’s recursive update loop (per `trm.md`) with a compact transformer core.
   - inside the core, use the same 3:1 kda-to-full-attention layer ratio.
   - use deep supervision steps and recursive refinement to align with trm’s strength on arc-agi.
17. **why**: `trm.md` shows trm’s strengths on arc-agi with small models; kda should preserve efficiency at longer contexts.

## training plan (both routes, separate pipelines)
18. **stage 0 sanity runs**: use small subsets for throughput sanity, loss stability, and tokenization checks (guided by `smol-llm.md` eval-first principles).
19. **stage 1 pretraining**: train from scratch to a token budget appropriate for 100m params.
20. **stage 2 continued pretraining**: add reasoning-heavy data and structured tasks (explicitly “mid-training” per `smol-llm.md`).
21. **stage 3 specialization**: arc-agi focused data; lower lr; smaller batch; aim for accuracy boost.

## post-training (on-policy distillation)
22. **on-policy distillation**: use a stronger teacher to generate outputs on arc-like prompts; distill on-policy as described in `smol-llm.md` (ensure tokenizer compatibility).
23. **why**: `smol-llm.md` highlights on-policy distillation as cost-effective for smaller models.

## evals and instrumentation
24. **eval suite setup** (eval-before-training, per `smol-llm.md`):
   - arc-agi dev split, arc-agi-2 public, sudoku/maze/logic tasks.
   - add lightweight language benchmarks for regression checks (short-form qa + simple code).
25. **automate evals**: use periodic checkpoints and a minimal eval runner; `smol-llm.md` emphasizes automated evals.

## hardware routes (choose per budget/availability)
26. **local rtx 3060 (dev only)**: run smoke tests, data sanity checks, and short notebook experiments. keep runs small to avoid tying up the workstation.
27. **kaggle t4 x2 (ddp learning run)**: use a small subset and short schedules to practice ddp mechanics without risking the main training.
28. **kaggle p100 (main training)**: single-gpu full runs for both route a and route b; simplest stable option for long training.
29. **tpu v5e-8 (defer)**: only consider if kda/trm kernels are confirmed tpu-compatible.
30. **why**: `ultrascale-playbook.md` stresses throughput vs. memory tradeoffs and systematic benchmarking.

## run checklists (concrete)
31. **ddp learning run (kaggle t4 x2)**:
   - verify dataset snapshot tag and tokenizer version.
   - start with short run (<= 5k steps) and log throughput + loss.
   - confirm gradient sync and identical loss curves across ranks.
   - save a checkpoint and resume once to validate recovery.
32. **single-gpu main run (kaggle p100)**:
   - pin dataset snapshot tag and config hash.
   - run a 1-epoch smoke phase before full token budget.
   - validate eval runner on a single checkpoint.
   - enable periodic checkpointing and logging.

## deliverables in this repo
33. **plan doc only**: document all steps, assumptions, and checkpoints in `plan.md`.
34. **if we proceed later**: create notebooks for each pipeline and staged training; keep pipelines separate except shared data prep.
