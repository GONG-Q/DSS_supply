# Open Science Appendix Draft

## Artifact Availability

We provide an anonymized research artifact that contains the core implementation, prompt files, representative visual outputs, execution logs, and a minimal runnable example for the intervention pipeline studied in this paper. The package is organized to support inspection, partial rerunning, and minimal repeatability checks without requiring access to private infrastructure. Third-party model checkpoints are not redistributed and must be obtained separately by the reviewer in accordance with their original licenses.

## What The Artifact Supports

The artifact is designed to support a minimal end-to-end verification run and artifact inspection. In particular, it includes:

- the code used for the intervention logic and related components;
- a self-contained minimal example in notebook and script form;
- cached intermediate statistics and sanitized execution logs for the minimal run;
- prompt files and representative output images used for qualitative inspection; and
- environment and launch scripts that document the expected software setup.

The included `run_minimal.sh` command reproduces the smallest complete execution path in the package once a local Stable Diffusion checkpoint path is provided. The optional `run_full.sh` command applies the same script-based pipeline to a larger prompt file.

## Reproducibility Scope

The artifact targets repeatability of the provided pipeline rather than redistribution of all external dependencies. Because diffusion outputs can vary across hardware, drivers, and library versions, minor numerical or visual differences may occur across systems. The package therefore includes cached intermediate outputs, sanitized logs, and parameter snapshots to make the executed steps transparent and auditable.

## Anonymization And Safety

To preserve double-blind review, local absolute paths and direct personal identifiers were removed from the shared code and visible logs where possible. The minimal example defaults to saving only the intervened output image, since the pre-intervention image may contain sensitive visual content. Reviewers can still inspect parameter settings, cached statistics, and execution traces through the accompanying metadata files.

## Access Conditions

All code and documentation in the package are released under the included license. External model weights remain subject to their own licenses and distribution terms. If the paper is accepted, the authors can release a camera-ready archival package and, where licensing permits, a public repository version aligned with the final manuscript.
