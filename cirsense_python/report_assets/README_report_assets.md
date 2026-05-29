# Report Figure Assets

This folder collects a small set of figures prepared for the course report.
The original experiment outputs remain in `outputs/`; these files are only
report-ready copies or combined summaries.

Recommended figures:

1. `fig1_system_pipeline.png`
   - Workflow of the contactless Wi-Fi sensing system.
   - Use in the "project goal and technical scheme" section.

2. `fig2_bpm_feature_examples.png`
   - CIR magnitude, respiration waveform, respiration spectrum, and time-frequency map.
   - Use to show preprocessing and feature extraction.

3. `fig3_multitarget_distance_profile.png`
   - Multi-target distance profile with detected peaks.
   - Use to show the formal multi-target distance extension.

4. `fig4_distance_method_comparison.png`
   - MAE, Accuracy, and Macro-F1 under strict distance classification.
   - Use in the distance estimation results section.

5. `fig5_bpm_method_comparison.png`
   - MAE, Accuracy, and Macro-F1 for respiration rate estimation methods.
   - Use in the respiration monitoring results section.

6. `fig6_confusion_matrices.png`
   - Test-set confusion matrices for BPM and strict distance classification.
   - Use in the model comparison section.

The figure text is English because the current Matplotlib runtime lacks a
Chinese font. Chinese captions can be added directly in the Word report.
