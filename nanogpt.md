# Repository Guidelines

## Project Structure & Module Organization
- `train.py`, `model.py`, and `sample.py` hold the core training, model, and sampling logic.
- `config/` contains runnable config presets (e.g., `config/train_shakespeare_char.py`).
- `data/` has dataset prep scripts and readmes (e.g., `data/shakespeare_char/prepare.py`).
- `assets/` stores images used by documentation.
- `bench.py` and `configurator.py` support benchmarking and CLI config parsing.

## Build, Test, and Development Commands
- `python data/shakespeare_char/prepare.py` downloads/prepares a tiny dataset for quick runs.
- `python train.py config/train_shakespeare_char.py` trains a small character-level model.
- `python sample.py --out_dir=out-shakespeare-char` remember to point at a trained run output.
- `torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py` runs DDP training.
- `python bench.py` profiles the core training loop.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and PEP 8-style naming (`snake_case` functions/vars).
- Configs are Python files in `config/` and should be named by task (e.g., `train_*.py`).
- Keep training flags explicit and readable; add comments only for non-obvious logic.

## Testing Guidelines
- No dedicated test suite is included; validate changes via quick training or sampling runs.
- Prefer minimal smoke runs on `data/shakespeare_char` before larger experiments.
- If you add tests, document how to run them in this file.

## Commit & Pull Request Guidelines
- Commit messages are short and descriptive; prefixes like `fix:` appear in history.
- If relevant, note dataset/config changes and training impacts in the PR description.
- Include sample outputs or loss curves when changing model or training behavior.

## Configuration & Safety Tips
- Default training uses `torch.compile()`; disable with `--compile=False` if unsupported.
- Use `--device=cpu` or `--device=mps` for non-CUDA environments.
- Keep outputs organized via `--out_dir` (e.g., `out-shakespeare-char`).
