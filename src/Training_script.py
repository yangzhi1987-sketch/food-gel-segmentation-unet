import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import segmentation_models_pytorch as smp

# ✅ Folder paths (your structure)
resized_img_dir = "/Users//Protein_Gel_AI_Project/confocal_dataset/train/resized_images"
resized_mask_dir = "/Users/Protein_Gel_AI_Project/confocal_dataset/train/resized_masks"
aug_img_dir = "/Users//Protein_Gel_AI_Project/confocal_dataset/train/augmented_images"
aug_mask_dir = "/Users/Documents/Protein_Gel_AI_Project/confocal_dataset/train/augmented_masks"
save_model_dir = "/Users/zhiy/Documents/Protein_Gel_AI_Project/confocal_dataset/saved_models"
plot_dir = "/Users/zhiy/Documents/Protein_Gel_AI_Project/confocal_dataset/results/plots"

# ✅ Transformation
transform = transforms.Compose([transforms.ToTensor()])

# ✅ Custom Dataset class
class ConfocalDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
            mask = (mask > 0.5).float()
        return img, mask

# ✅ Collect resized data (up to gel_99 only)
resized_img_paths = []
resized_mask_paths = []
for i in range(1, 100):
    img_name = f"gel_{i:02d}.jpg"
    mask_name = f"gel_{i:02d}_fiji_mask.png"
    img_path = os.path.join(resized_img_dir, img_name)
    mask_path = os.path.join(resized_mask_dir, mask_name)
    if os.path.exists(img_path) and os.path.exists(mask_path):
        resized_img_paths.append(img_path)
        resized_mask_paths.append(mask_path)

# ✅ Collect ~1000 augmented data
aug_img_files = sorted([f for f in os.listdir(aug_img_dir) if f.endswith(".jpg")])[:1000]
aug_img_paths = [os.path.join(aug_img_dir, f) for f in aug_img_files]
aug_mask_paths = [os.path.join(aug_mask_dir, f.replace(".jpg", ".png")) for f in aug_img_files]

# ✅ Merge all valid pairs
all_image_paths = resized_img_paths + aug_img_paths
all_mask_paths = resized_mask_paths + aug_mask_paths

# ✅ Shuffle the data
combined = list(zip(all_image_paths, all_mask_paths))
random.shuffle(combined)
all_image_paths, all_mask_paths = zip(*combined)

# ✅ Create dataset and DataLoaders
dataset = ConfocalDataset(list(all_image_paths), list(all_mask_paths), transform=transform)
val_split = 0.1
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
train_data, val_data = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
val_loader = DataLoader(val_data, batch_size=4, shuffle=False)

# ✅ Model: U-Net with pretrained ResNet34 encoder
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
    activation=None
).to(device)

# ✅ Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ✅ Training loop
num_epochs = 300
train_losses, val_losses = [], []
best_val_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            val_loss += loss.item()
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    # ✅ Save best model
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        timestamp = datetime.now().strftime("%Y-%m-%d")
        best_model_path = os.path.join(save_model_dir, f"unet_resnet34_300epoches_{timestamp}.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Best model saved at: {best_model_path}")

# ✅ Plot and save loss curve
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_plot_path = os.path.join(plot_dir, f"loss_curve_{datetime.now().strftime('%Y-%m-%d')}.png")
plt.savefig(loss_plot_path)
plt.show()
print(f"📊 Loss curve saved at: {loss_plot_path}")