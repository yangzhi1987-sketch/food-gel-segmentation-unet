#!/usr/bin/env python3
"""
Inference for U-Net (ResNet34 encoder) on CLSM images.

Usage:
  python src/infer_unet_resnet34.py     --images data/example_images     --weights models/unet_resnet34_model_300epochs_best_2025-03-26.pth     --out results/pred_masks     --resize 256 256     --threshold 0.5     --save-overlay

Notes:
- Expects single-channel grayscale inputs (will convert if needed).
- By default resizes inputs to 256x256 to match training; change with --resize.
- Outputs binary masks as PNG with the same basename as input.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# --- Model definition: U-Net with ResNet34 encoder via segmentation-models-pytorch
try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("Please install segmentation-models-pytorch: pip install segmentation-models-pytorch", file=sys.stderr)
    raise

def get_model(in_channels=1, classes=1):
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes,
        activation=None
    )

def load_grayscale(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img

def make_overlay(gray_img, mask_bin):
    # gray to 3-channel
    vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    # red overlay for mask
    overlay = vis.copy()
    overlay[mask_bin > 0] = (0, 0, 255)
    # alpha blend
    out = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder of input CLSM images")
    ap.add_argument("--weights", required=True, help="Path to .pth model weights")
    ap.add_argument("--out", required=True, help="Folder to save predicted masks")
    ap.add_argument("--resize", nargs=2, type=int, default=[256, 256], help="Resize H W (default 256 256)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold (default 0.5)")
    ap.add_argument("--save-overlay", action="store_true", help="Also save color overlays")
    args = ap.parse_args()

    in_dir = Path(args.images)
    out_dir = Path(args.out)
    ov_dir = out_dir.parent / "pred_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlay:
        ov_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(in_channels=1, classes=1).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()
    sigm = nn.Sigmoid()

    img_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    files = [p for p in sorted(in_dir.iterdir()) if p.suffix.lower() in img_exts]
    if not files:
        print(f"No images found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    H, W = args.resize
    with torch.no_grad():
        for p in files:
            img = load_grayscale(p)
            img_resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            x = img_resized.astype(np.float32) / 255.0
            x = (x - 0.5) / 0.5  # simple standardization (match training if applicable)
            x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
            logits = model(x)
            prob = sigm(logits).cpu().numpy()[0, 0]
            mask = (prob >= args.threshold).astype(np.uint8) * 255

            out_path = out_dir / f"{p.stem}.png"
            cv2.imwrite(str(out_path), mask)

            if args.save_overlay:
                overlay = make_overlay(img_resized, mask)
                cv2.imwrite(str(ov_dir / f"{p.stem}.png"), overlay)

            print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
