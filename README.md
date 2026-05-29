# CIRSense Python: Contactless Wi-Fi Sensing

This repository contains a Python implementation for a course project on
contactless Wi-Fi sensing with the CIRSense CSI dataset. The project estimates
respiration rate and target distance from real 802.11ax CSI measurements, and
compares classical machine learning methods with deep learning and
physics-guided residual models.

The implementation is original course-project code. It refers to the CIRSense
paper and dataset, but it is not a copy of the authors' MATLAB/GitHub code.

## Highlights

- CSI to CIR preprocessing for Wi-Fi sensing.
- Denoising, normalization, dynamic path alignment, FFT/STFT features.
- Respiration rate estimation and single-target distance estimation.
- NLOS respiration and multi-target distance smoke/full experiments.
- Classical baselines: PCA+SVM and PCA+KNN.
- Deep learning methods: CNN, CNN-LSTM, robust DNN ensemble, multiview fusion.
- Physics-guided residual correction for distance estimation.
- Strict distance classification mode to avoid overly optimistic coarse labels.
- Report-ready figures in `cirsense_python/report_assets/`.

## Repository Layout

```text
.
├─ README.md
├─ .gitignore
├─ cirsense_python/
│  ├─ README.md
│  ├─ requirements.txt
│  ├─ main.py
│  ├─ run_experiments.py
│  ├─ learning_experiments.py
│  ├─ learning_data.py
│  ├─ models.py
│  ├─ train.py
│  ├─ baselines.py
│  ├─ cirsense_core.py
│  ├─ csi_to_cir.py
│  ├─ preprocessing.py
│  ├─ feature_extraction.py
│  ├─ multitarget_distance.py
│  ├─ result_audit.py
│  └─ report_assets/
```

Generated outputs, preprocessing caches, local Office files, and the raw
CIRSense dataset are intentionally ignored by Git.

## Dataset

This repository does **not** include the CIRSense dataset. Please download or
obtain the dataset from its original source and place it locally as:

```text
cirsense_python/
└─ cirsense-dataset-real-world-80211ax-csi-measurements-wireless-sensing/
   └─ CIRSense_dataset/
      ├─ breathe/
      ├─ distance/
      ├─ multitarget/
      └─ nlos_breathe/
```

Do not upload third-party datasets to GitHub unless the dataset license
explicitly permits redistribution. For a public repository, it is usually safer
to provide download instructions and keep the dataset out of Git.

## Paper and Data Attribution

This project is based on the research direction and dataset associated with:

> Ruiqi Kong and He Chen, "CIRSense: Rethinking WiFi Sensing with Channel
> Impulse Response," arXiv:2510.11374, 2025.
> DOI: https://doi.org/10.48550/arXiv.2510.11374

The CIRSense paper, dataset, and any original author-provided materials remain
the intellectual property of their respective authors. This repository only
contains independent Python course-project code and report assets. If you use
the original CIRSense dataset or paper, please cite the original authors and
follow the license or access terms from the dataset source.

Suggested BibTeX:

```bibtex
@misc{kong2025cirsense,
  title = {CIRSense: Rethinking WiFi Sensing with Channel Impulse Response},
  author = {Kong, Ruiqi and Chen, He},
  year = {2025},
  eprint = {2510.11374},
  archivePrefix = {arXiv},
  primaryClass = {eess.SP},
  doi = {10.48550/arXiv.2510.11374}
}
```

## Installation

```bash
cd cirsense_python
pip install -r requirements.txt
```

After dependencies and data are installed, experiments can run offline.

## Quick Test

```bash
python main.py --task bpm --max-files 3 --run-name quick_bpm_test --save-plots
python main.py --task distance --max-files 3 --run-name quick_distance_test --save-plots
```

Quick tests are written to `cirsense_python/output_test/`.

## Core Experiments

```bash
python run_experiments.py --suite core_full --run-name core_full_final --resume
```

This runs:

- BPM estimation on `breathe/`
- distance estimation on `distance/`
- NLOS BPM estimation on `nlos_breathe/`
- multi-target distance estimation on `multitarget/distance/`

## Learning Experiments

Strict distance classification:

```bash
python learning_experiments.py --run-name distance_strict_final --source-run-name full_experiment_final_comparison --tasks distance --methods physics_baseline pca_svm pca_knn dnn_robust_ensemble physics_residual physics_residual_ensemble multiview_residual --epochs 120 --physics-run-name full_experiment_final --distance-class-mode strict
```

BPM learning with the legacy CIRSense dynamic-path waveform:

```bash
python learning_experiments.py --run-name deep_stronger_v4 --source-run-name full_experiment_final_comparison --tasks bpm distance --methods dnn_robust_ensemble --epochs 120 --bpm-signal-mode legacy
```

BPM signal modes:

- `legacy`: original CIRSense dynamic-path waveform; this is the default and
  produced the strongest BPM results in the current report.
- `quality`: newer multi-antenna/multi-tap candidate search using respiration
  spectrum quality; useful for ablation and future work.

See `cirsense_python/README.md` for detailed commands and output descriptions.

## Recommended Report Results

Distance estimation under strict class labels:

| Method | MAE / m | Acc | Macro-F1 |
| --- | ---: | ---: | ---: |
| PCA+SVM | 0.192 | 1.000 | 1.000 |
| PCA+KNN | 0.170 | 0.889 | 0.863 |
| DNN robust ensemble | 0.086 | 1.000 | 1.000 |
| Multiview residual | 0.024 | 0.981 | 0.988 |
| Physics residual ensemble | 0.022 | 0.981 | 0.988 |

Respiration rate estimation:

| Method | MAE / bpm | Acc | Macro-F1 |
| --- | ---: | ---: | ---: |
| DNN robust ensemble | 0.442 | 0.825 | 0.838 |
| Multiview fusion | 0.521 | 0.850 | 0.875 |
| Augmented multiview | 0.526 | 0.900 | 0.699 |

## Notes on Reproducibility

- Use file-level train/validation/test splitting to avoid window-level leakage.
- Do not include true distance labels as model input features.
- Use `--distance-class-mode strict` for final distance Acc/F1 reporting.
- Keep raw data and generated outputs outside Git; regenerate them locally.

## License

No open-source license has been selected yet. If you plan to publish this
repository publicly, add a license file that matches your intended reuse policy.
The CIRSense dataset and paper remain governed by their own authors' terms.
