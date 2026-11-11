# Lunar Crater Detection and Apollo Seismology

Research code and derived results for a course project by Aniket Pandurang Raval, Department of Civil Engineering, Indian Institute of Technology Madras. This is a working research repository; the manuscript and figures are not a journal-ready release.

## Scope

This repository contains a YOLO11n crater-detection experiment on the public NASA–IBM SomBench NAC benchmark, two classical image-processing comparisons, and exploratory Apollo lunar seismic analyses. The test split contains 46 image tiles and 5,934 retained crater annotations. The project does not claim that YOLO11n is state of the art, derive lunar surface ages, or establish a repeating moonquake source.

## Current results

Ultralytics test evaluation reported precision 0.706, recall 0.669, mAP50 0.720, and mAP50–95 0.314. In a separate fixed-confidence (0.25), one-to-one IoU ≥ 0.5 comparison on the same filtered test set, YOLO11n had precision 0.774, recall 0.548, and F1 0.642. Hough-circle and Laplacian-of-Gaussian baselines had F1 0.0065 and 0.0111. Both classical settings selected a validation-grid boundary, so these results compare only the tested configurations. See the JSON files in `results/`.

Four artificial-impact records at Apollo 14 station S14 were screened for missing data and instrument-rail clipping. Apollo 15 S-IVB has upper-rail samples; Apollo 17 S-IVB has a long data gap. The KO-26/KO-40 cross-correlation is exploratory, uses catalogue-time windows rather than picked P arrivals, and does not confirm or refute published repeating-event findings.

## Reproduce

Use a GPU-enabled Kaggle notebook or compatible environment. Install `requirements.txt`, download SomBench NAC from [NASA–IBM AI4Science](https://huggingface.co/datasets/nasa-ibm-ai4science/Sombench-NAC-Crater-Detection), place the dataset under `work/data/Sombench-NAC-Crater-Detection/`, then run the scripts in order: `download_sombench_nac.py` if using its downloader, `prepare_sombench_yolo.py`, and `train_sombench_yolo.py`. The training script writes outputs under `work/runs/`. The Apollo scripts require the public Apollo PSE data; see `data/README.md`.

Reported hardware/software: Kaggle, Python 3.13.16, PyTorch 2.11.0+cu130, Ultralytics 8.4.205, Tesla T4. See `config/kaggle_run.json` for settings. Training and validation/test protocols are not site-held-out.

## Data and licensing

SomBench NAC is distributed under CC BY 4.0 by its publisher. Apollo data are available from the NASA PDS Geosciences Node. This repository does not contain the full benchmark dataset. Ultralytics licensing and all source/data terms must be reviewed before a public research release. No code license or DOI has been assigned yet; this repository should remain private until license review, author review, and figure cleanup are complete.

## Figure status

Figures under `figures/` are diagnostic outputs from the experiment. Some contain in-image titles and are not prepared as final journal figures.
