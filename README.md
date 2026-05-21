**NCRN**

**TDL CHECKING - 64×64 Array**

TDL Validation · Channel Estimation · Massive MIMO_

Documentation

## Overview

This package contains the Stage II TDL validation pipeline for the paper "Noise-Conditioned Residual Learning for Robust Channel Estimation in High-Mobility Massive MIMO." All training scripts are hard-coded to 64×64 - no command-line arguments needed.

NCRN is benchmarked against LS, MMSE, and Sa-DLCS across 11 3GPP TDL scenarios spanning InH, UMi, UMa, RMa, and HST environments, over an SNR range of −15 dB to +20 dB.

## Methods

| **Method** | **Type**      | **Description**                                   |
| ---------- | ------------- | ------------------------------------------------- |
| LS         | Classical     | Least Squares estimation                          |
| MMSE       | Classical     | Minimum Mean Square Error estimation              |
| Sa-DLCS    | Deep Learning | Sparse deep learning channel estimator (baseline) |
| NCRN       | Deep Learning | Proposed neural channel refinement network        |

_CNN-LSTM and CNN-Sparse are trained for completeness but excluded from the final NMSE figures for visual clarity. Sa-DLCS and NCRN are the primary comparison pair alongside LS and MMSE baselines._

## Repository Structure

MMO_PACKAGE_64/

├── stage_tdl/ # Stage II - TDL training scripts

│ ├── cnn_lstm/train_cnn_lstm_tdl.py

│ ├── cnn_sparse/train_cnn_sparse_tdl.py

│ ├── sa_dlcs/train_sa_dlcs_tdl.py

│ └── ncrn/train_ncrn_tdl.py

├── shared/

│ ├── channels.py # TDL channel generation

│ ├── baselines.py # LS, MMSE, Sa-DLCS helpers

│ └── \__init_\_.py

├── models/ # Trained .keras + .pkl files

├── outputs/

│ ├── figures/ # PDF and PNG plots

│ └── csv/ # Raw NMSE results per SNR

├── plotting/plot_all_figures.py

├── train_all.sh

└── README.md

## Quick Start

\# Train all 4 TDL models (~20 min on M1)

cd MMO_PACKAGE_64

bash train_all.sh

\# Skip CNN variants, train only Sa-DLCS and NCRN

SKIP_CNN=1 bash train_all.sh

\# Generate all NMSE figures

python plotting/plot_all_figures.py

\# Fast preview (4× fewer samples)

python plotting/plot_all_figures.py --quick

\# Skip per-scenario panels

python plotting/plot_all_figures.py --skip scenarios

## TDL Scenarios

All 10 3GPP TDL scenarios evaluated, grouped by propagation condition:

| **Group** | **Environment** | **Scenario**    | **Speed (km/h)** |
| --------- | --------------- | --------------- | ---------------- |
| LOS       | InH             | InH-LOS         | 3                |
| LOS       | UMi             | UMi-Street-LOS  | 30               |
| LOS       | UMa             | UMa-LOS         | 30               |
| LOS       | RMa             | RMa-LOS         | 120              |
| NLOS      | InH             | InH-NLOS        | 3                |
| NLOS      | UMi             | UMi-Street-NLOS | 30               |
| NLOS      | UMa             | UMa-NLOS        | 30               |
| NLOS      | RMa             | RMa-NLOS        | 120              |
| O2I       | UMi             | UMi-O2I         | 3                |
| HST       | RMa             | RMa-HST         | 300              |

## Trained Models

| **File**                | **Algorithm** | **Channel** | **Used in figures**                |
| ----------------------- | ------------- | ----------- | ---------------------------------- |
| cnn_lstm_tdl_64.keras   | CNN-LSTM      | TDL Stage   | Trained only - excluded from plots |
| cnn_sparse_tdl_64.keras | CNN-Sparse    | TDL Stage   | Trained only - excluded from plots |
| sa_dlcs_tdl_64.keras    | Sa-DLCS       | TDL Stage   | All 5 NMSE figures                 |
| ncrn_tdl_64.keras       | NCRN          | TDL Stage   | All 5 NMSE figures                 |

## Output Figures

| **File**                      | **Description**                                    |
| ----------------------------- | -------------------------------------------------- |
| nmse_tdl_64.pdf / .png        | Averaged NMSE across all 10 TDL scenarios          |
| tdl_scenarios_LOS.pdf / .png  | Per-scenario NMSE panels - LOS group (4 subplots)  |
| tdl_scenarios_NLOS.pdf / .png | Per-scenario NMSE panels - NLOS group (4 subplots) |
| tdl_scenarios_O2I.pdf / .png  | Per-scenario NMSE panels - O2I group (2 subplots)  |
| tdl_scenarios_HST.pdf / .png  | Per-scenario NMSE panels - HST group (1 subplot)   |

_All figures saved as both PDF (vector, print-ready) and PNG (600 dpi). CSVs with raw NMSE values (dB and linear) are saved to outputs/csv/._

## Per-Script Usage

\# Train each model individually (run from repo root)

cd stage2_tdl/cnn_lstm

python train_cnn_lstm_tdl.py # → models/cnn_lstm_tdl_64.keras

cd stage2_tdl/cnn_sparse

python train_cnn_sparse_tdl.py # → models/cnn_sparse_tdl_64.keras

cd stage2_tdl/sa_dlcs

python train_sa_dlcs_tdl.py # → models/sa_dlcs_tdl_64.keras

cd stage2_tdl/ncrn

python train_ncrn_tdl.py # → models/ncrn_tdl_64.keras

## Requirements

pip install tensorflow numpy matplotlib

| **Package** | **Tested Version** | **Notes**             |
| ----------- | ------------------ | --------------------- |
| Python      | 3.9 - 3.11         |                       |
| TensorFlow  | 2.12 - 2.16        | Metal build for M1/M2 |
| NumPy       | 1.24+              |                       |
| Matplotlib  | 3.7+               |                       |

## Memory and Runtime Notes

| **Task**                                   | **Apple M1** | **RTX 3090** |
| ------------------------------------------ | ------------ | ------------ |
| Training - all 4 models                    | ~20 min      | ~8 min       |
| Plotting - full (N_tdl=500, N_scen=1000)   | ~30 min      | ~12 min      |
| Plotting - --quick (N_tdl=125, N_scen=250) | ~8 min       | ~3 min       |

Default batch sizes: 8 for CNN-LSTM / CNN-Sparse / Sa-DLCS, 4 for NCRN.

## Troubleshooting

### "Model not found" warnings during plotting

The script will skip that figure cleanly. Run the missing training step first, or use --skip:

\# Skip averaged TDL plot, run only per-scenario panels

python plotting/plot_all_figures.py --skip stage2

\# Skip per-scenario panels, run only averaged TDL plot

python plotting/plot_all_figures.py --skip scenarios

### TensorFlow not detecting GPU on M1

Make sure you are using the Metal-enabled build:

pip install tensorflow-metal

### Import errors from shared/

Always run plotting from the repo root, not from inside plotting/:

\# Correct

python plotting/plot_all_figures.py

\# Wrong - will fail to import shared/

cd plotting && python plot_all_figures.py