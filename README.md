# SqzAct

This repository presents the source code for the paper **"SqzAct: Taming Activation Outliers for Efficient 4-bit LLM Inference via Block-Level Squeezing"**, accepted by **EMSOFT 2026**.

## Directory Structure

```
SqzAct/
├── SqzAct_algo/     # Core quantization algorithm implementation
├── simulator/        # Hardware accelerator simulator (based on DNNWeaver2 & BitFusion)
└── README.md
```

- **`SqzAct_algo/`** — Contains the quantization algorithm, model wrappers, QwT compensation, and evaluation scripts for LLMs and Vision Transformers.
- **`simulator/`** — A cycle-accurate simulator for systolic-array accelerators supporting mixed-precision computation, used to evaluate hardware performance.

## Supported Models

LLaMA family, Qwen2/2.5, Qwen3, Mistral, OPT, and DeepSeek-R1-Distill-Qwen. Model sizes range from 0.5B to 30B parameters.
