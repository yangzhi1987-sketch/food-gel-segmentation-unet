# Inference and Metrics Scripts

**How to use**

1) Create the environment
```bash
conda env create -f environment.yml
conda activate foodgel
```

2) Run inference
```bash
python src/infer_unet_resnet34.py   --images data/example_images   --weights models/unet_resnet34_model_300epochs_best_2025-03-26.pth   --out results/pred_masks   --resize 256 256   --threshold 0.5   --save-overlay
```

3) Compute metrics and statistics
```bash
python src/metrics_analysis.py   --gt data/example_masks   --pred results/pred_masks   --out results/metrics   --invert 0
```

Outputs:
- `results/metrics/metrics_per_image.csv`
- `results/metrics/metrics_summary.csv`
- `results/metrics/metrics_stats_tests.csv`
- `results/metrics/bland_altman_plots.png`
