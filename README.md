# food-gel-segmentation-unet
Deep learning framework (U-Net with ResNet34 encoder) for segmentation and quantitative analysis of confocal images of plant protein gels.
# Model Weights

The trained model weights for **ResNet34 U-Net** (300 epochs) are hosted on Zenodo.

🧩 **Download from Zenodo**  
 DOI: 10.5281/zenodo.17463487

### Model Information
- Architecture: U-Net with ResNet34 encoder  
- Training images: CLSM plant protein gel samples covering soy, pea, quinoa, and faba bean protein gels.
- Epochs: 300  
- File: `unet_resnet34_model_300epochs_best_2025-03-26.pth`  

For inference, place the downloaded model in the `/models` directory and run:
```bash
python src/infer_unet_resnet34.py --weights models/unet_resnet34_model_300epochs_best_2025-03-26.pth
