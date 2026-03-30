# Turnitout Essay Humanizer

Interactive sentence-level paraphrase tool for academic essays, powered by a fine-tuned LoRA adapter on [Gemma-2-9B-IT](https://huggingface.co/mlx-community/gemma-2-9b-it-4bit) via Apple MLX.

For each sentence, the tool generates **3 diverse rewrite options**. You pick the best one (or keep the original). The result is a naturally rewritten essay with your choices baked in.

## Requirements

- **Apple Silicon** Mac (M1/M2/M3/M4) with **12 GB+ unified memory**
- Python 3.10+

## Quick Start

```bash
git clone https://github.com/kevin0818-lxd/turnitout-essay-humanizer.git
cd turnitout-essay-humanizer

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/interactive_rewrite.py \
  --input your_essay.txt \
  --output rewritten.txt
```

On **first run**, the base model (~5 GB) is automatically downloaded from Hugging Face. Subsequent runs use the cached model.

## How It Works

```
Input sentence
      │
      ▼
  ┌───────────────────┐
  │ Translate to 3    │   (diverse CN translations via varied prompts/temps)
  │ Chinese versions  │
  └───────┬───────────┘
          │
          ▼
  ┌───────────────────┐
  │ Translate each CN │   (LoRA-adapted Gemma-2-9B for CN→EN)
  │ back to English   │
  └───────┬───────────┘
          │
          ▼
  ┌───────────────────┐
  │ User picks 1 of 3 │   (or keeps original)
  │ options            │
  └────────────────────┘
```

The **CN-bridge strategy** produces naturally diverse outputs because different Chinese phrasings lead to structurally different English sentences — far more varied than direct paraphrase sampling.

## Usage

```
python scripts/interactive_rewrite.py --input essay.txt --output out.txt [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Path to UTF-8 English essay |
| `--output` | *(required)* | Output path |
| `--model` | `mlx-community/gemma-2-9b-it-4bit` | Base model (HF ID or local path) |
| `--adapter` | `adapter/` | LoRA adapter directory |
| `--generation-mode` | `cn2en-bridge` | `cn2en-bridge` (recommended) or `direct` |
| `--state` | *(none)* | JSON checkpoint file for resume |
| `--temperature` | `0.65` | Base sampling temperature |
| `--max-tokens` | `512` | Max tokens per generation |
| `--auto` | *(none)* | Non-interactive: always pick 0/1/2/3 |
| `--cn-boundary-paras` | *(none)* | Insert CN boundary blocks at paragraph edges |

### Resume from checkpoint

If you quit mid-session, pass `--state progress.json` to save/resume your place:

```bash
python scripts/interactive_rewrite.py \
  --input essay.txt --output out.txt \
  --state progress.json
```

### Non-interactive mode

Auto-select option 1 for every sentence (no user input required):

```bash
python scripts/interactive_rewrite.py \
  --input essay.txt --output out.txt --auto 1
```

## Project Structure

```
turnitout-essay-humanizer/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── adapter/
│   ├── adapters.safetensors    # Fine-tuned LoRA weights (39 MB)
│   └── adapter_config.json
└── scripts/
    └── interactive_rewrite.py  # Main CLI
```

## License

[MIT](LICENSE)
