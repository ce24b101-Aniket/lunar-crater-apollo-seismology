# 🌕 Lunar Crater Detection and Apollo Seismology

A course research project using computer vision to detect lunar craters and Apollo seismic records to explore lunar impacts and seismic signals.

**Author:** Aniket Pandurang Raval  
**Department:** Civil Engineering, Indian Institute of Technology Madras  
**Project status:** Working research repository; results are preliminary and have not been peer reviewed.

## Project overview

This project combines two analyses:

1. **Crater detection:** Fine-tune YOLO11n on the NASA–IBM SomBench NAC benchmark and compare it with two classical image-processing baselines.
2. **Apollo seismology:** Check four artificial-impact records for data gaps and clipping, and compare two catalogued Apollo seismic-event windows.

The analyses are presented together as a research workflow. The crater and seismic results have **not** been combined into a validated geological interpretation.

## Current results
### Site-held-out robustness check

We trained a fresh YOLO11n while holding out every tile from the Highland site. Apollo 17 tiles were used for validation. On the 73 Highland test tiles (7,763 retained crater labels), Ultralytics reported mAP50 of 0.704 and mAP50–95 of 0.246. At fixed confidence 0.25 and IoU 0.5, YOLO11n achieved precision 0.669, recall 0.663, and F1 0.666.

On the same test tiles, Hough circles achieved F1 0.0106 and LoG achieved F1 0.0188. Hough’s selected parameter and LoG’s selected threshold were both at the edge of their validation search ranges, so these are limited baseline comparisons. This is an exploratory check: results from the original benchmark split had already been inspected. It does not establish performance across all lunar sites or state-of-the-art status.

Details: [`split_manifest.json`](results/split_manifest.json), [`ultralytics_test_metrics.json`](results/ultralytics_test_metrics.json), [`yolo_fixed_confidence.json`](results/yolo_fixed_confidence.json), [`hough_validation_and_test.json`](results/hough_validation_and_test.json), and [`log_validation_and_test.json`](results/log_validation_and_test.json).
### Crater detection

The benchmark test split used **46 image tiles** with **5,934 retained crater annotations**.

| Evaluation | Model or method | Precision | Recall | F1 | mAP50 | mAP50–95 |
|---|---|---:|---:|---:|---:|---:|
| Ultralytics test evaluator | YOLO11n | 0.706 | 0.669 | — | 0.720 | 0.314 |
| Fixed confidence 0.25, IoU ≥ 0.5 | YOLO11n | 0.774 | 0.548 | 0.642 | — | — |
| Same fixed-threshold comparison | Hough circles | 0.0044 | 0.0125 | 0.0065 | — | — |
| Same fixed-threshold comparison | Laplacian of Gaussian (LoG) | 0.0073 | 0.0229 | 0.0111 | — | — |

The evaluator metrics and fixed-confidence results use different evaluation procedures, so they are reported separately. Hough and LoG thresholds were selected on the validation set; both selected values were at the edge of the tested threshold grid. These results compare the tested configurations only. **They do not establish that YOLO11n is state of the art or that classical methods generally perform poorly.**

Detailed results are in [`results/`](results/).

### Apollo seismic analysis

Four artificial-impact records at Apollo 14 station S14 were checked for missing samples and instrument clipping:

- Apollo 14 LM: 6.27% missing; maximum gap of about 1.23 seconds.
- Apollo 15 S-IVB: 6.96% missing; maximum gap of about 18.74 seconds, with samples at the upper instrument rail.
- Apollo 16 S-IVB: 6.25% missing; maximum gap of one sample.
- Apollo 17 S-IVB: 32.18% missing; maximum gap of about 421 seconds.

The KO-26/KO-40 comparison produced a cross-correlation of 0.527 at a lag of −0.547 seconds. The windows were based on catalogue times, not picked P arrivals, and no control pairs were run. This exploratory result **does not confirm or refute** the published repeating-event interpretation.

## Data

The full datasets are not stored in this repository.

- **SomBench NAC:** [NASA–IBM AI4Science dataset](https://huggingface.co/datasets/nasa-ibm-ai4science/Sombench-NAC-Crater-Detection), distributed under CC BY 4.0.
- **Apollo Passive Seismic Experiment:** [NASA PDS Geosciences Node archive](https://doi.org/10.17189/9ykc-er91). The Apollo network DOI is [10.7914/SN/XA_1969](https://doi.org/10.7914/SN/XA_1969).
- Data retrieval notes are in [`data/README.md`](data/README.md).

## Reproducing the crater experiment

The reported run used Kaggle with Python 3.13.16, PyTorch 2.11.0+cu130, Ultralytics 8.4.205, and a Tesla T4 GPU. The training configuration used 640-pixel images, batch size 4, seed 42, and up to 20 epochs.

From the repository root, run:

```bash
python -m pip install -r requirements.txt
python src/download_sombench_nac.py
python src/prepare_sombench_yolo.py
python src/train_sombench_yolo.py
