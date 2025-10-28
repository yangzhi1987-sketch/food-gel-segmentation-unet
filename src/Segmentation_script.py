import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.metrics import jaccard_score, f1_score
from matplotlib.backends.backend_pdf import PdfPages

# ✅ Paths
test_img_dir = "/Users/Documents/Protein_Gel_AI_Project/confocal_dataset/test/resized_images"
test_mask_dir = "/Users/Documents/Protein_Gel_AI_Project/confocal_dataset/test/resized_masks"
model_path = "/Users/Documents/Protein_Gel_AI_Project/confocal_dataset/saved_models/unet_resnet34_300epoches_2025-03-27.pth"
save_dir = "/Users/zhiy/Documents/Protein_Gel_AI_Project/confocal_dataset/results/test_metrics"
predicted_mask_dir = os.path.join(save_dir, "predicted_masks")
os.makedirs(predicted_mask_dir, exist_ok=True)

# ✅ Load model
import segmentation_models_pytorch as smp
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ✅ Transform
transform = transforms.Compose([transforms.ToTensor()])

# ✅ Results containers
results = []
pdf_path = os.path.join(save_dir, "side_by_side_comparison.pdf")
pdf = PdfPages(pdf_path)

# ✅ Helper functions
def porosity(mask): return 100 * (1.0 - np.mean(mask))
def aggregation(mask): return 100 * np.mean(mask)
def fractal_dimension(mask):
    from skimage.measure import label
    def boxcount(Z, k):
        S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                            np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where(S > 0)[0])
    Z = mask.astype(bool)
    p = min(Z.shape)
    n = 2 ** np.floor(np.log2(p))
    sizes = 2 ** np.arange(int(np.log2(n)), 1, -1)
    counts = [boxcount(Z, size) for size in sizes]
    return -np.polyfit(np.log(sizes), np.log(counts), 1)[0]

# ✅ Loop through test images
for fname in sorted(os.listdir(test_img_dir)):
    if not fname.endswith(".jpg"):
        continue
    base = fname.replace(".jpg", "")
    img_path = os.path.join(test_img_dir, fname)
    mask_path = os.path.join(test_mask_dir, f"{base}_test_mask.png")
    pred_path = os.path.join(predicted_mask_dir, f"{base}_pred.png")

    if not os.path.exists(mask_path):
        print(f"❌ Missing mask: {mask_path}")
        continue

    # Load image and mask
    image = Image.open(img_path).convert("L")
    mask = Image.open(mask_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_binary = (pred > 0.5).astype(np.uint8) * 255

    # Save prediction
    cv2.imwrite(pred_path, pred_binary)

    # Metrics
    mask_np = np.array(mask) > 127
    pred_np = pred_binary > 127
    iou = jaccard_score(mask_np.flatten(), pred_np.flatten())
    dice = f1_score(mask_np.flatten(), pred_np.flatten())

    results.append({
        "Image": fname,
        "IoU": iou,
        "Dice": dice,
        "GT_Porosity": porosity(mask_np),
        "Pred_Porosity": porosity(pred_np),
        "GT_Aggregation": aggregation(mask_np),
        "Pred_Aggregation": aggregation(pred_np),
        "GT_Fractal": fractal_dimension(mask_np),
        "Pred_Fractal": fractal_dimension(pred_np)
    })

    # ✅ Show and Save Side-by-Side
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original")
    ax[1].imshow(mask_np, cmap='gray')
    ax[1].set_title("Ground Truth")
    ax[2].imshow(pred_np, cmap='gray')
    ax[2].set_title("AI Prediction")
    for a in ax: a.axis('off')
    plt.suptitle(f"{fname} | IoU: {iou:.2f} | Dice: {dice:.2f}")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.show()  # ✅ Show in notebook

pdf.close()
print("✅ All side-by-side comparisons saved and shown!")

# ✅ DataFrame + CSV
df = pd.DataFrame(results)
df.to_csv(os.path.join(save_dir, "test_metrics_summary.csv"), index=False)

# ✅ IoU and Dice Bar Charts
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Image", y="IoU", color="steelblue")
plt.xticks(rotation=90)
plt.title("IoU Scores")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Image", y="Dice", color="darkorange")
plt.xticks(rotation=90)
plt.title("Dice Scores")
plt.tight_layout()
plt.show()