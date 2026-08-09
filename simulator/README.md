# SqzAct Simulator

This directory contains the hardware accelerator simulator for **SqzAct**, adapted from [ANT-Quantization](https://github.com/clevercool/ANT-Quantization/tree/main/ant_simulator) and built on top of **DNNWeaver2** and **BitFusion**. The simulator models a systolic-array accelerator with configurable mixed-precision support to evaluate the hardware efficiency of Block-Level Squeezing against competing designs.

## Prerequisites

- Ubuntu 18.04+ LTS
- Python 3.8+
- gcc 7.5+
- Anaconda / Miniconda

## Setup

1. Create a conda environment and install dependencies:
```bash
conda create -n sqzact_sim python=3.8
conda activate sqzact_sim
pip install -r requirements.txt
```

2. Clone and compile CACTI for SRAM energy modeling:
```bash
git clone https://github.com/HewlettPackard/cacti ./bitfusion/sram/cacti/
make -C ./bitfusion/sram/cacti/
```

## Quick Start

Run the full simulation pipeline comparing MANT, OliVe, SPARK, and SqueezeAct across all benchmarks:

```bash
python run.py
```

This generates `results/my_res.csv` containing normalized cycle counts and energy breakdowns.

## Hardware Configuration

Architecture parameters are defined in `.ini` files under `hardwareConf/`. Each file specifies systolic array dimensions (`N x M`), supported precision range (`high_prec` / `low_prec`), SRAM sizes, and memory bandwidth. To add a new configuration, create a new `.ini` file following the existing format.

## Evaluation

The simulation results (`results/my_res.csv`) include:
- **Cycles** — Normalized to MANT baseline, with geomean speedups across all benchmarks.
- **Energy** — Breakdown into Static, DRAM, Buffer, and Core components, normalized to MANT totals.

As shown below, the `./results/my_res.csv` provides the template. You can fill it to generate Figure 9 and 10 in the paper.
