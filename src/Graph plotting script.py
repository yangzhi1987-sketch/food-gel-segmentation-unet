import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# === Load your results CSV ===
csv_path = "/Users/zhiy/Documents/Protein_Gel_AI_Project/confocal_dataset/full_structure_analysis_results.csv"
df = pd.read_csv(csv_path)

# === Set figure save path ===
fig_output_path = "/Users/zhiy/Documents/Protein_Gel_AI_Project/confocal_dataset/summary_figures"
os.makedirs(fig_output_path, exist_ok=True)

# === Define plotting functions ===
def correlation_subplot(ax, x, y, xlabel, ylabel, title):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    reg = LinearRegression().fit(x, y)
    y_pred = reg.predict(x)
    r2 = r2_score(y, y_pred)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    sign = '+' if intercept >= 0 else '-'
    eqn = f"y = {slope:.2f}x {sign} {abs(intercept):.2f}\nR² = {r2:.3f}"
    ax.scatter(x, y, color='blue')
    ax.plot(x, y_pred, 'r-', label=eqn)
    ax.plot([min(x), max(x)], [min(x), max(x)], 'k--', linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

def bar_subplot(ax, labels, gt_values, ai_values, ylabel, title):
    x_vals = np.arange(len(labels))
    width = 0.35
    ax.bar(x_vals - width/2, gt_values, width, label="GT", color='skyblue')
    ax.bar(x_vals + width/2, ai_values, width, label="AI", color='salmon')
    ax.set_xticks(x_vals)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

# === Figure 1: Correlation Plots ===
fig1, axs1 = plt.subplots(2, 2, figsize=(12, 10))
correlation_subplot(axs1[0, 0], df["GT_Porosity_%"], df["AI_Porosity_%"], "GT Porosity (%)", "AI Porosity (%)", "1A. Porosity")
correlation_subplot(axs1[0, 1], df["GT_Aggregation_%"], df["AI_Aggregation_%"], "GT Aggregation (%)", "AI Aggregation (%)", "1B. Aggregation")
correlation_subplot(axs1[1, 0], df["GT_Fractal_D"], df["AI_Fractal_D"], "GT Fractal D", "AI Fractal D", "1C. Fractal Dimension")
correlation_subplot(axs1[1, 1], df["GT_Lacunarity"], df["AI_Lacunarity"], "GT Lacunarity", "AI Lacunarity", "1D. Lacunarity")
fig1.tight_layout()
fig1.show()
fig1.savefig(os.path.join(fig_output_path, "Figure1_Correlation_Plots.png"), dpi=300)
fig1.savefig(os.path.join(fig_output_path, "Figure1_Correlation_Plots.pdf"))

# === Figure 2: Bar Charts ===
labels = df["Sample"]
fig2, axs2 = plt.subplots(2, 2, figsize=(14, 10))
bar_subplot(axs2[0, 0], labels, df["GT_Porosity_%"], df["AI_Porosity_%"], "Porosity (%)", "2A. Porosity (GT vs AI)")
bar_subplot(axs2[0, 1], labels, df["GT_Aggregation_%"], df["AI_Aggregation_%"], "Aggregation (%)", "2B. Aggregation (GT vs AI)")
bar_subplot(axs2[1, 0], labels, df["GT_Fractal_D"], df["AI_Fractal_D"], "Fractal Dimension", "2C. Fractal Dimension (GT vs AI)")
bar_subplot(axs2[1, 1], labels, df["GT_Lacunarity"], df["AI_Lacunarity"], "Lacunarity", "2D. Lacunarity (GT vs AI)")
fig2.tight_layout()
fig2.show()
fig2.savefig(os.path.join(fig_output_path, "Figure2_Bar_Charts.png"), dpi=300)
fig2.savefig(os.path.join(fig_output_path, "Figure2_Bar_Charts.pdf"))

# === Save caption text ===
caption_text = """
Figure 1. Correlation plots comparing AI-predicted vs ground truth (GT) values for:
(A) Porosity, (B) Protein Aggregation Area, (C) Fractal Dimension, and (D) Lacunarity.
Each panel includes a linear regression line (red), the y=x reference (dashed), and the corresponding R² value.

Figure 2. Bar charts comparing GT and AI values across gel samples for:
(A) Porosity, (B) Protein Aggregation Area, (C) Fractal Dimension, and (D) Lacunarity.
Each metric is plotted for gel01 to gel23 with side-by-side bars for GT and AI results.
"""
with open(os.path.join(fig_output_path, "Figure_Captions.txt"), "w") as f:
    f.write(caption_text)

print("✅ All figures and captions saved to:", fig_output_path)
