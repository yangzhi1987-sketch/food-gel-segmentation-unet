import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from skimage.util import view_as_blocks

# === Folder paths ===
base_path = "/Users/Protein_Gel_AI_Project/confocal_dataset/"
gt_folder = os.path.join(base_path, "test/resized_masks/")
ai_folder = os.path.join(base_path, "results/test_metrics/predicted_masks/")
results_folder = base_path
os.makedirs(results_folder, exist_ok=True)

# === Load binary mask ===
def load_mask(path):
    return (np.array(Image.open(path).convert("L")) > 128).astype(np.uint8)

# === Compute porosity and aggregation ===
def compute_porosity_and_aggregation(binary_mask):
    total_pixels = binary_mask.size
    protein_pixels = np.sum(binary_mask == 1)
    porosity = 100 * (total_pixels - protein_pixels) / total_pixels
    aggregation = 100 - porosity
    return round(porosity, 2), round(aggregation, 2)

# === Compute fractal dimension ===
def compute_fractal_dimension(binary_img, box_sizes=[4, 8, 16, 32, 64]):
    h, w = binary_img.shape
    counts = []
    for box_size in box_sizes:
        h_trim = h - h % box_size
        w_trim = w - w % box_size
        trimmed = binary_img[:h_trim, :w_trim]
        blocks = view_as_blocks(trimmed, block_shape=(box_size, box_size))
        non_empty = np.any(blocks, axis=(2, 3))
        counts.append(np.sum(non_empty))
    log_eps = np.log(1 / np.array(box_sizes, dtype=np.float64))
    log_counts = np.log(counts)
    slope, _ = np.polyfit(log_eps, log_counts, 1)
    return round(slope, 4)

# === Compute lacunarity ===
def compute_lacunarity(binary_img, box_sizes=[4]):
    lacunarities = []
    for box_size in box_sizes:
        h, w = binary_img.shape
        h_trim = h - h % box_size
        w_trim = w - w % box_size
        trimmed = binary_img[:h_trim, :w_trim]
        reshaped = trimmed.reshape(h_trim // box_size, box_size, w_trim // box_size, box_size)
        blocks = reshaped.sum(axis=(1, 3)).astype(np.float64)
        mean = np.mean(blocks)
        std_dev = np.std(blocks)
        if mean > 0:
            lac = (std_dev / mean) ** 2 + 1
            lacunarities.append((box_size, round(lac, 4)))
    return lacunarities[0][1] if lacunarities else np.nan

# === Collect results ===
gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(".png")])
ai_files = sorted([f for f in os.listdir(ai_folder) if f.endswith(".png")])
results = []

for idx, (gt_file, ai_file) in enumerate(zip(gt_files, ai_files)):
    sample_id = f"gel{str(idx+1).zfill(2)}"
    gt_mask = load_mask(os.path.join(gt_folder, gt_file))
    ai_mask = load_mask(os.path.join(ai_folder, ai_file))

    poro_gt, aggr_gt = compute_porosity_and_aggregation(gt_mask)
    poro_ai, aggr_ai = compute_porosity_and_aggregation(ai_mask)
    fd_gt = compute_fractal_dimension(gt_mask)
    fd_ai = compute_fractal_dimension(ai_mask)
    lac_gt = compute_lacunarity(gt_mask)
    lac_ai = compute_lacunarity(ai_mask)

    results.append({
        "Sample": sample_id,
        "GT_Porosity_%": poro_gt,
        "AI_Porosity_%": poro_ai,
        "GT_Aggregation_%": aggr_gt,
        "AI_Aggregation_%": aggr_ai,
        "GT_Fractal_D": fd_gt,
        "AI_Fractal_D": fd_ai,
        "GT_Lacunarity": lac_gt,
        "AI_Lacunarity": lac_ai
    })

# === Save to CSV ===
df = pd.DataFrame(results)
csv_path = os.path.join(results_folder, "full_structure_analysis_results.csv")
df.to_csv(csv_path, index=False)
print(f"✅ Results saved to: {csv_path}")

# === Plot functions ===
def plot_correlation(df, col_gt, col_ai, ylabel, title, filename):
    x = df[col_gt].values.reshape(-1, 1)
    y = df[col_ai].values
    reg = LinearRegression().fit(x, y)
    y_pred = reg.predict(x)
    r2 = r2_score(y, y_pred)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    sign = '+' if intercept >= 0 else '-'
    equation = f"y = {slope:.2f}x {sign} {abs(intercept):.2f} (R² = {r2:.3f})"

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, color='blue', label="Samples")
    plt.plot(x, y_pred, 'r-', label=equation)
    plt.plot([min(x), max(x)], [min(x), max(x)], 'k--', label="y = x")
    plt.xlabel(col_gt.replace("_", " "))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, filename))
    plt.close()

def plot_bar(df, col_gt, col_ai, ylabel, title, filename):
    labels = df["Sample"]
    x_vals = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(12, 5))
    plt.bar(x_vals - width/2, df[col_gt], width, label="GT", color='skyblue')
    plt.bar(x_vals + width/2, df[col_ai], width, label="AI", color='salmon')
    plt.xticks(x_vals, labels, rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, filename))
    plt.close()

# === Plot all metrics ===
plot_correlation(df, "GT_Porosity_%", "AI_Porosity_%", "AI Porosity (%)", "Porosity Correlation: GT vs AI", "correlation_porosity.png")
plot_bar(df, "GT_Porosity_%", "AI_Porosity_%", "Porosity (%)", "GT vs AI Porosity (Per Sample)", "bar_porosity.png")

plot_correlation(df, "GT_Aggregation_%", "AI_Aggregation_%", "AI Aggregation (%)", "Aggregation Correlation: GT vs AI", "correlation_aggregation.png")
plot_bar(df, "GT_Aggregation_%", "AI_Aggregation_%", "Aggregation (%)", "GT vs AI Aggregation (Per Sample)", "bar_aggregation.png")

plot_correlation(df, "GT_Fractal_D", "AI_Fractal_D", "AI Fractal Dimension", "Fractal Dimension Correlation: GT vs AI", "correlation_fractal.png")
plot_bar(df, "GT_Fractal_D", "AI_Fractal_D", "Fractal Dimension", "GT vs AI Fractal Dimension (Per Sample)", "bar_fractal.png")

plot_correlation(df, "GT_Lacunarity", "AI_Lacunarity", "AI Lacunarity", "Lacunarity Correlation: GT vs AI", "correlation_lacunarity.png")
plot_bar(df, "GT_Lacunarity", "AI_Lacunarity", "Lacunarity", "GT vs AI Lacunarity (Per Sample)", "bar_lacunarity.png")
