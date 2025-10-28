#!/usr/bin/env python3
"""
Compute GT vs AI metrics and statistical comparisons.

Outputs:
- Per-image CSV with IoU, Dice, Porosity %, Aggregation %, Fractal Dimension, Lacunarity
- Summary CSV with means, std, biases (AI - GT)
- Bland–Altman plots (PNG) for each metric
- Paired t-test and Wilcoxon signed-rank results (printed and saved)

Usage:
  python src/metrics_analysis.py     --gt data/example_masks     --pred results/pred_masks     --out results/metrics     --invert 0

Notes:
- By default masks are assumed "protein = white (255), pores = black (0)".
- Porosity % = fraction of black pixels * 100.
- Aggregation % = fraction of white pixels * 100 = 100 - Porosity %.
- If your convention is flipped, set --invert 1 to invert masks before analysis.
- Fractal Dimension and Lacunarity are computed on the protein (white) component.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ---------- Helper functions ----------

def binarize(img, invert=False):
    """Ensure binary 0/255; optionally invert."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    if invert:
        bw = 255 - bw
    return bw

def iou_score(gt, pr):
    gt_b = gt > 0
    pr_b = pr > 0
    inter = np.logical_and(gt_b, pr_b).sum()
    union = np.logical_or(gt_b, pr_b).sum()
    return inter / union if union > 0 else 0.0

def dice_score(gt, pr):
    gt_b = gt > 0
    pr_b = pr > 0
    inter = np.logical_and(gt_b, pr_b).sum()
    denom = gt_b.sum() + pr_b.sum()
    return (2.0 * inter / denom) if denom > 0 else 0.0

def porosity_and_aggregation(mask):
    """Assume protein white=255, pores black=0. Return (porosity%, aggregation%)."""
    total = mask.size
    pores = (mask == 0).sum()
    porosity = 100.0 * pores / total if total > 0 else 0.0
    aggregation = 100.0 - porosity
    return porosity, aggregation

def fractal_dimension_boxcount(mask, min_box=2, max_box=None, n_scales=8):
    """Box-counting dimension on protein (white) component."""
    bw = mask > 0
    H, W = bw.shape
    if max_box is None:
        max_box = min(H, W) // 2
    sizes = np.unique(np.logspace(np.log2(min_box), np.log2(max_box), num=n_scales, base=2, dtype=int))
    sizes = sizes[sizes > 1]
    if len(sizes) < 2:
        return np.nan

    counts = []
    for s in sizes:
        newH = int(np.ceil(H / s) * s)
        newW = int(np.ceil(W / s) * s)
        pad = np.zeros((newH, newW), dtype=bool)
        pad[:H, :W] = bw
        reshaped = pad.reshape(newH // s, s, newW // s, s)
        blocks = reshaped.any(axis=(1, 3))
        counts.append(np.count_nonzero(blocks))

    x = np.log(1.0 / sizes.astype(np.float64))
    y = np.log(np.array(counts, dtype=np.float64) + 1e-8)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)

def lacunarity_gliding_box(mask, min_box=2, max_box=None, n_scales=8):
    """
    Estimate lacunarity using a gliding-box method on the protein (white) component.
    Higher lacunarity = more heterogeneity in void/cluster sizes.
    Returns average lacunarity across scales for a compact scalar.
    """
    bw = (mask > 0).astype(np.uint8)
    H, W = bw.shape
    if max_box is None:
        max_box = min(H, W) // 2
    sizes = np.unique(np.logspace(np.log2(min_box), np.log2(max_box), num=n_scales, base=2, dtype=int))
    sizes = sizes[sizes > 1]
    if len(sizes) < 1:
        return np.nan

    lac_vals = []
    ii = cv2.integral(bw)
    for s in sizes:
        rows = H - s + 1
        cols = W - s + 1
        if rows <= 0 or cols <= 0:
            continue
        # compute all s x s window sums using integral image
        sums = np.empty((rows, cols), dtype=np.int32)
        for y in range(rows):
            y0, y1 = y, y + s
            for x in range(cols):
                x0, x1 = x, x + s
                sums[y, x] = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]

        mean_m = sums.mean()
        var_m = sums.var()
        if mean_m <= 1e-8:
            continue
        Lambda = var_m / (mean_m ** 2) + 1.0
        lac_vals.append(Lambda)

    if not lac_vals:
        return np.nan
    return float(np.mean(lac_vals))

def bland_altman(ax, gt_vals, ai_vals, title, ylabel):
    gt_vals = np.asarray(gt_vals, dtype=np.float64)
    ai_vals = np.asarray(ai_vals, dtype=np.float64)
    mean_vals = (gt_vals + ai_vals) / 2.0
    diff = ai_vals - gt_vals
    md = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loa_low = md - 1.96 * sd
    loa_high = md + 1.96 * sd

    ax.scatter(mean_vals, diff, s=22)
    ax.axhline(md, linestyle="--")
    ax.axhline(loa_low, linestyle=":")
    ax.axhline(loa_high, linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("Mean of GT and AI")
    ax.set_ylabel(ylabel)

    return md, sd, loa_low, loa_high

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="Folder with GT masks")
    ap.add_argument("--pred", required=True, help="Folder with AI predicted masks")
    ap.add_argument("--out", required=True, help="Output folder (CSV + plots)")
    ap.add_argument("--invert", type=int, default=0, help="Set to 1 to invert masks before analysis")
    args = ap.parse_args()

    gt_dir = Path(args.gt)
    pr_dir = Path(args.pred)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    names = sorted([p.name for p in pr_dir.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg",".tif",".tiff")])
    if not names:
        print(f"No predicted masks found in {pr_dir}")
        return

    for name in names:
        gt_p = gt_dir / name
        pr_p = pr_dir / name
        if not gt_p.exists():
            gt_cands = list(gt_dir.glob(f"{Path(name).stem}.*"))
            if not gt_cands:
                print(f"Warning: GT not found for {name}, skipping.")
                continue
            gt_p = gt_cands[0]

        gt_img = cv2.imread(str(gt_p), cv2.IMREAD_GRAYSCALE)
        pr_img = cv2.imread(str(pr_p), cv2.IMREAD_GRAYSCALE)
        if gt_img is None or pr_img is None:
            print(f"Warning: failed to read one of {gt_p} or {pr_p}, skipping.")
            continue

        gt_bw = binarize(gt_img, invert=bool(args.invert))
        pr_bw = binarize(pr_img, invert=bool(args.invert))

        # Metrics
        iou = iou_score(gt_bw, pr_bw)
        dice = dice_score(gt_bw, pr_bw)
        por_gt, agg_gt = porosity_and_aggregation(gt_bw)
        por_ai, agg_ai = porosity_and_aggregation(pr_bw)
        fd_gt = fractal_dimension_boxcount(gt_bw)
        fd_ai = fractal_dimension_boxcount(pr_bw)
        lac_gt = lacunarity_gliding_box(gt_bw)
        lac_ai = lacunarity_gliding_box(pr_bw)

        rows.append({
            "image": name,
            "IoU": iou, "Dice": dice,
            "Porosity_GT": por_gt, "Porosity_AI": por_ai,
            "Aggregation_GT": agg_gt, "Aggregation_AI": agg_ai,
            "FractalDim_GT": fd_gt, "FractalDim_AI": fd_ai,
            "Lacunarity_GT": lac_gt, "Lacunarity_AI": lac_ai
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    per_image_csv = out_dir / "metrics_per_image.csv"
    df.to_csv(per_image_csv, index=False)
    print(f"Saved per-image metrics: {per_image_csv}")

    # Summary and biases
    summary = {
        "IoU_mean": df["IoU"].mean(),
        "IoU_std": df["IoU"].std(),
        "Dice_mean": df["Dice"].mean(),
        "Dice_std": df["Dice"].std(),
        "Bias_Porosity": (df["Porosity_AI"] - df["Porosity_GT"]).mean(),
        "Bias_Aggregation": (df["Aggregation_AI"] - df["Aggregation_GT"]).mean(),
        "Bias_FractalDim": (df["FractalDim_AI"] - df["FractalDim_GT"]).mean(),
        "Bias_Lacunarity": (df["Lacunarity_AI"] - df["Lacunarity_GT"]).mean(),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "metrics_summary.csv", index=False)

    # Paired tests for each structural metric
    stats_rows = []
    for gt_col, ai_col, label in [
        ("Porosity_GT", "Porosity_AI", "Porosity"),
        ("Aggregation_GT", "Aggregation_AI", "Aggregation"),
        ("FractalDim_GT", "FractalDim_AI", "FractalDim"),
        ("Lacunarity_GT", "Lacunarity_AI", "Lacunarity"),
    ]:
        x = df[gt_col].values
        y = df[ai_col].values
        # paired t-test
        t_stat, p_t = stats.ttest_rel(y, x, nan_policy="omit")
        # Wilcoxon signed-rank
        diffs = y - x
        try:
            w_stat, p_w = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            w_stat, p_w = np.nan, np.nan
        stats_rows.append({"Metric": label, "Paired_t_stat": t_stat, "Paired_t_p": p_t, "Wilcoxon_stat": w_stat, "Wilcoxon_p": p_w})

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out_dir / "metrics_stats_tests.csv", index=False)
    print(f"Saved stats: {out_dir / 'metrics_stats_tests.csv'}")

    # Bland–Altman plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    pairs = [
        ("Porosity", df["Porosity_GT"].values, df["Porosity_AI"].values, "AI - GT Porosity (%)"),
        ("Aggregation", df["Aggregation_GT"].values, df["Aggregation_AI"].values, "AI - GT Aggregation (%)"),
        ("Fractal Dimension", df["FractalDim_GT"].values, df["FractalDim_AI"].values, "AI - GT Fractal Dimension"),
        ("Lacunarity", df["Lacunarity_GT"].values, df["Lacunarity_AI"].values, "AI - GT Lacunarity"),
    ]
    for ax, (title, gtv, aiv, ylabel) in zip(axes.ravel(), pairs):
        bland_altman(ax, gtv, aiv, f"Bland–Altman: {title}", ylabel)
    plt.tight_layout()
    ba_path = out_dir / "bland_altman_plots.png"
    plt.savefig(ba_path, dpi=300)
    plt.close()
    print(f"Saved Bland–Altman plots: {ba_path}")

if __name__ == "__main__":
    main()
