# SqzAct Algorithm

This directory contains the core implementation of **SqzAct: Block-Level Squeezing** for efficient 4-bit LLM inference. It supports post-training quantization (PTQ) with per-channel activation clamping, compensation, and perplexity / zero-shot evaluation across multiple LLM families.

## Setup

1. Create a conda environment and install dependencies:
```bash
conda create -n sqzact python=3.10
conda activate sqzact
pip install -r requirements.txt
```

2. Modify `model_path` in `mycode/llm.py:167` to point to your model directory. Models should be in HuggingFace format. For example:
```text
localssd/lbxj/Meta-Llama-3-8B/
├── config.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── model.safetensors.index.json
├── tokenizer.json
└── tokenizer_config.json
```

## Quick Start

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python mycode/lm_eval_reasoning_task.py --model_name=llama3_8B_hf --task=arc_easy --eval_clamp_qwt
```

## Evaluation Modes

| Flag | Description |
|---|---|
| `--eval_base` | Evaluate base (unquantized) model |
| `--eval_quant` | W4A8 quantized (no clamp) |
| `--eval_clamp` | W4A8 with clamp quantization |
| `--eval_clamp_qwt` | W4A8 with clamp quantization + compensation |
| `--train_clamp_qwt` | Fine-tune compensat parameters |

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `LLaMA-2-7B` | Model name (see supported list) |
| `--task` | `wikitext` | Evaluation dataset: `wikitext` or `c4`, `arc_easy`, `arc_challenge`, `hellaswag`, `piqa`, `winogrande`, `boolq` |
| `--wgt_nbit` | `4` | Weight bit-width |
| `--act_nbit` | `8` | Activation bit-width |


## Supported Models

TinyLlama-1.1B-Chat, LLaMA-2-7B/13B, LLaMA-3-8B, Qwen2.5-{0.5B, 1.5B, 7B, 14B}, OPT-{125m, 1.3B, 2.7B, 6.7B, 13B}, DeepSeek-R1-Distill-Qwen-{1.5B, 7B}, Mistral-7B.

## Supported Datasets

- **PPL**: `wikitext` (wikitext-2-raw-v1), `c4`
- **Zero-shot**: `arc_easy`, `arc_challenge`, `hellaswag`, `piqa`, `winogrande`, `boolq`

