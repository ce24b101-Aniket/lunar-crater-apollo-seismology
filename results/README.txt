Site-held-out crater analysis results

The split holds all Highland tiles out of training and validation. Apollo 17 is validation; Highland is the 73-tile test site. This is an exploratory robustness check because results from the original benchmark split had already been inspected.

YOLO11n test evaluator: precision 0.710, recall 0.648, mAP50 0.704, mAP50-95 0.246. At fixed confidence 0.25 and one-to-one IoU >= 0.5: precision 0.669, recall 0.663, F1 0.666. These are separate evaluation protocols.

On the same held-out site, Hough circles had F1 0.0106 and LoG had F1 0.0188. Both classical settings were selected at a validation-grid boundary, so they are limited tested configurations. This single-site comparison does not establish state-of-the-art performance or broad model superiority.

No checkpoint, raw benchmark data, or Apollo waveforms are included in this results bundle.
