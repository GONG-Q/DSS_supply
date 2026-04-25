# Open Science Checklist

This checklist is prepared for artifact packaging, review, and camera-ready open-science disclosure.

## Availability

- Code is included in `DSS_Code/` and `sexual_eraser_example/`.
- Minimal runnable entry points are included in `run_minimal.sh` and `sexual_eraser_example/effective_erae_nudity_demo.ipynb`.
- A script-based batch entry point is included in `run_full.sh`.
- Example outputs, cached intermediate statistics, and execution logs are included in `sexual_eraser_example/demo_outputs/`.
- Text prompt datasets used by the artifact are included in `Datasets/`.
- Representative visualization results are included in `Erasure_Results/`.

## Environment

- A reproducible software environment is described in `environment.yml`.
- Python package requirements are also listed in `DSS_Code/requirements.txt`.
- The artifact expects the reviewer to provide a local Stable Diffusion checkpoint path via `SD_MODEL_PATH`.
- Model weights are not redistributed in this package because of upstream licensing and size constraints.

## Reproduction Workflow

- Minimal verification command: `bash run_minimal.sh`
- Optional larger batch command: `bash run_full.sh`
- The minimal path reuses cached stage-1 statistics when they already exist in `sexual_eraser_example/demo_outputs/`.
- The artifact stores sanitized runtime metadata in `run_config.json` and `execution_log.txt`.

## Documentation

- High-level artifact instructions are documented in `README.md`.
- A paper-to-artifact mapping table is included in `README.md`.
- A paper appendix draft is included in `OpenScience_Appendix.md`.

## Anonymity And Safety

- Local absolute paths in shared code and visible logs were replaced by placeholders or repository-relative paths.
- The minimal runnable unit defaults to saving intervened outputs only.
- Python bytecode caches should not be included in submission bundles.
- Reviewers should avoid adding personal absolute paths into committed configs or notebooks before submission.

## Known Limitations

- Exact numerical and visual outputs can vary across GPUs, CUDA versions, and upstream diffusion-library versions.
- The artifact is intended for repeatability of the included pipeline, not for redistribution of third-party checkpoints.
- Some example prompts and outputs concern sensitive-content suppression; the package therefore favors sanitized logs and limited visual exposure.
