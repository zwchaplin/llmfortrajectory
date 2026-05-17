# Repository Guidelines

## Project Structure & Module Organization

This repository is a small Python codebase for LLM-based trajectory prediction experiments. Core scripts live at the repository root:

- `training.py` fine-tunes a model with LoRA adapters.
- `inference.py` runs validation inference with a base model and adapter checkpoint.
- `evaluation.py` evaluates prediction JSON files.
- `create_data_split.py` prepares train/validation data from GPT-Driver-style raw data.
- `prompt_message.py` contains prompt construction helpers shared by the experiment scripts.

The `data/` directory contains `train_with_token.json` and `val_with_token.json`. Generated artifacts such as checkpoints, inference outputs, and evaluation results should go under `checkpoints/`, `output/`, and `results/` respectively; create these directories locally as needed.

## Build, Test, and Development Commands

Create and activate the documented environment:

```bash
conda create -n llmtp python=3.9
conda activate llmtp
pip install -r requirements.txt
```

Run the main workflows with the provided wrappers:

```bash
./run_training.sh      # train LoRA adapter into checkpoints/
./run_inference.sh     # write predictions into output/
./run_evaluation.sh    # evaluate JSON predictions into results/
```

On Windows PowerShell, call the Python scripts directly with the same arguments shown in the `.sh` files.

## Coding Style & Naming Conventions

Use Python 3.9-compatible code. Follow PEP 8 conventions: 4-space indentation, `snake_case` for functions, variables, and file names, and descriptive CLI argument names such as `--validation_data_file`. Keep experiment configuration explicit in command-line arguments rather than hidden globals. Prefer small helper functions when logic is shared between training, inference, and evaluation.

## Testing Guidelines

There is currently no committed automated test suite. Before submitting changes, run the affected script on a small or existing data file and confirm it completes. For evaluation changes, verify `python evaluation.py --prediction_file <file> --output_path <file>` creates valid JSON. If adding tests, place them under `tests/`, name files `test_*.py`, and use `pytest`.

## Commit & Pull Request Guidelines

Recent history uses short, imperative summaries such as `Updated README: description.` Keep commit titles concise and specific, preferably under 72 characters. Pull requests should include a brief purpose statement, commands run, expected data/checkpoint requirements, and sample output paths. Link related issues or papers when relevant. Do not commit large model checkpoints, generated result dumps, credentials, or local IDE settings.

## Security & Configuration Tips

Some Hugging Face models require access approval or tokens. Keep tokens in your shell environment or local configuration, never in source files. Treat downloaded checkpoints and generated outputs as local artifacts unless they are intentionally small and reproducible.
