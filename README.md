# CCS-Style Artifact Overview

This directory is organized as an anonymized artifact package for inspection, minimal rerunning, and open-science disclosure. The fastest path is the script entry point `run_minimal.sh`, which executes the cleaned minimal example in `sexual_eraser_example/`.

## At A Glance

- Minimal runnable entry: `run_minimal.sh`
- Optional larger batch entry: `run_full.sh`
- Minimal notebook entry: `sexual_eraser_example/effective_erae_nudity_demo.ipynb`
- Minimal script entry: `sexual_eraser_example/effective_erae_nudity.py`
- Environment specification: `environment.yml`
- Open-science materials: `OpenScience_Checklist.md`, `OpenScience_Appendix.md`, `LICENSE`

## Recommended Reviewer Workflow

1. Create the environment from `environment.yml`.
2. Export a local diffusion checkpoint path: `export SD_MODEL_PATH=/path/to/model`.
3. Run `bash run_minimal.sh`.
4. Inspect `sexual_eraser_example/demo_outputs/` for the result image, cached statistics, execution log, and run metadata.
5. Optionally run `bash run_full.sh` for a larger prompt file.

## Directory Structure

- `DSS_Code/`: main implementation snapshot and configuration files for the broader method
- `Datasets/`: prompt files used for different tasks and concepts
- `Erasure_Results/`: representative visualization results for several concept-erasure settings
- `sexual_eraser_example/`: minimal runnable example, notebook, prompt file, caches, logs, and provided outputs
- `OpenScience_Checklist.md`: artifact and disclosure checklist
- `OpenScience_Appendix.md`: paper-ready appendix draft
- `LICENSE`: artifact license
- `environment.yml`: environment specification
- `run_minimal.sh`: smallest end-to-end command
- `run_full.sh`: optional larger batch command

## Minimal Verification Unit

The minimal verification unit is centered on:

- `sexual_eraser_example/effective_erae_nudity.py`
- `sexual_eraser_example/effective_erae_nudity_demo.ipynb`

Its design goals are:

- keep the execution path short and inspectable;
- avoid showing local absolute paths in shared outputs;
- save sanitized metadata and logs;
- default to saving only the intervened image in the minimal run; and
- reuse cached stage-1 statistics when available.

The default prompt file for the minimal script is:

- `sexual_eraser_example/minimal_prompts.txt`

The default minimal outputs are written to:

- `sexual_eraser_example/demo_outputs/`

Key provided files in that directory include:

- `result/single_result.png`
- `execution_log.txt`
- `run_config.json`
- `stage1_cache.pt`
- `stats/*.npy`

## Paper Claims To Artifact

| Paper claim or component | Artifact evidence |
| --- | --- |
| The method can be inspected as a two-stage intervention pipeline. | `sexual_eraser_example/effective_erae_nudity.py`, `DSS_Code/` |
| A minimal end-to-end run can be reproduced from model loading to final output. | `run_minimal.sh`, `sexual_eraser_example/effective_erae_nudity_demo.ipynb`, `sexual_eraser_example/demo_outputs/` |
| Stage-1 calibration statistics can be cached and reused. | `sexual_eraser_example/demo_outputs/stage1_cache.pt`, `execution_log.txt` |
| The artifact exposes intermediate statistics for inspection. | `sexual_eraser_example/demo_outputs/stats/*.npy`, `run_config.json` |
| Qualitative examples are included for manual inspection. | `sexual_eraser_example/intervened_provided_cases/`, `Erasure_Results/` |
| The software environment and launch procedure are documented. | `environment.yml`, `run_minimal.sh`, `run_full.sh`, `OpenScience_Checklist.md` |
| The package is prepared for open-science disclosure and double-blind review. | `OpenScience_Checklist.md`, `OpenScience_Appendix.md`, sanitized paths in the minimal example |

## Quick Start

### 1. Create The Environment

```bash
conda env create -f environment.yml
conda activate dss-artifact
```

### 2. Set The Model Path

```bash
export SD_MODEL_PATH=/path/to/stable-diffusion-checkpoint
```

The package does not redistribute model weights. Reviewers should use a locally available checkpoint that is compatible with the code.

### 3. Run The Minimal Example

```bash
bash run_minimal.sh
```

This command uses:

- the cleaned script entry in `sexual_eraser_example/effective_erae_nudity.py`;
- the default prompt file `sexual_eraser_example/minimal_prompts.txt`; and
- the default output directory `sexual_eraser_example/demo_outputs/`.

### 4. Optional Larger Batch Run

```bash
bash run_full.sh
```

By default, this uses `Datasets/nude_i2p.txt` as the prompt file and writes to `sexual_eraser_example/full_run_outputs/`.

If you want to also save pre-intervention images during the larger run:

```bash
SAVE_ORIGINAL=1 bash run_full.sh
```

## Environment Notes

Primary dependency references:

- `environment.yml`
- `DSS_Code/requirements.txt`

Main Python packages include:

- `torch`
- `torchvision`
- `diffusers`
- `transformers`
- `scikit-learn`
- `numpy`
- `matplotlib`
- `umap-learn`
- `Pillow`
- `tqdm`
- `pyyaml`
- `scipy`

## Anonymization Notes

This artifact is prepared for double-blind review:

- local absolute paths in visible code and logs were replaced by placeholders or repository-relative paths;
- the minimal script sanitizes path fields before writing `run_config.json`;
- the minimal path defaults to not saving original images; and
- reviewers should avoid packaging local caches such as `__pycache__/` or `.pyc` files.

## Open Science Materials

The package includes:

- `OpenScience_Checklist.md`
- `OpenScience_Appendix.md`
- `LICENSE`

These files are intended to support artifact evaluation, paper appendix drafting, and later camera-ready release preparation.
