# Satellite Image Based Flood Detection Using Deep Learning

A deep learning project that automatically identifies flooded areas from aerial images. Instead of manually inspecting disaster images one by one, the model looks at every pixel of an image and classifies what it belongs to — flooded buildings, flooded roads, water, trees, vehicles, and more. Based on this, it gives a simple final verdict on whether flood-related damage is present in the scene.

> **A small note on the data:** although the project title says "satellite", the model is actually trained on the **[FloodNet dataset](https://www.kaggle.com/datasets/imroze/floodnet-cropresized-512-wpartialmasks)**, which contains aerial/UAV images captured after Hurricane Harvey. So the model works on aerial imagery, not satellite imagery.

## Quick links

- 📂 **Dataset (Kaggle):** [FloodNet — Crop-Resized 512 with Partial Masks](https://www.kaggle.com/datasets/imroze/floodnet-cropresized-512-wpartialmasks)
- 📄 **FloodNet paper:** [FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding](https://arxiv.org/abs/2012.02951)
- 💻 **FloodNet official code (Bina Lab):** [github.com/BinaLab/FloodNet-Supervised_v1.0](https://github.com/BinaLab/FloodNet-Supervised_v1.0)
- 📄 **DeepLabV3 paper:** [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
- 📄 **MobileNetV3 paper:** [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

## Why I built this

After a flood, rescue and relief teams need to know *where* the damage is — which roads are blocked, which buildings are surrounded by water. Doing this manually from hundreds of aerial photos is slow. This project automates the first level of that analysis: feed in an aerial image, and the model maps the flooded regions and reports a verdict within seconds.

## What the model does

- Takes an aerial image as input
- Performs **semantic segmentation** — every pixel is classified into one of 10 classes
- Calculates how much of the scene is flooded infrastructure (flooded buildings + flooded roads)
- Gives a final verdict: **Flood detected / Uncertain / No flood**

The important design decision here: **plain water alone never triggers a flood verdict.** A river or a lake is natural water, not a disaster. The verdict is based only on flooded buildings and flooded roads, because those indicate actual damage.

## Dataset

- **FloodNet** — download from Kaggle: [imroze/floodnet-cropresized-512-wpartialmasks](https://www.kaggle.com/datasets/imroze/floodnet-cropresized-512-wpartialmasks)
- Original dataset and code by Bina Lab: [FloodNet-Supervised_v1.0 on GitHub](https://github.com/BinaLab/FloodNet-Supervised_v1.0) | [Research paper](https://arxiv.org/abs/2012.02951)
- Aerial images from Hurricane Harvey flood regions
- **398 usable images** after cleaning, split with zero overlap:
  - Training: 278 images
  - Validation: 59 images
  - Testing: 61 images
- **10 classes:** Background, Building-Flooded, Building-Non-Flooded, Road-Flooded, Road-Non-Flooded, Water, Tree, Vehicle, Pool, Grass
- Some masks are partial, so unlabelled pixels are handled using `ignore_index = 255` during training and evaluation

## Model and training

- **Architecture:** DeepLabV3 with a MobileNetV3-Large backbone (chosen to keep the model light and fast)
- **ASPP dilation rates:** 12, 24, 36
- **Total parameters:** 11,022,650 (about 11 million)
- **Input size:** 256 × 256 (images are natively 512 × 512)
- **Optimizer:** Adam, learning rate 5e-4, with cosine annealing
- **Loss function:** Weighted Cross-Entropy (to handle class imbalance) + Masked Dice Loss
- **Batch size:** 4 | **Seed:** 42
- **Augmentation:** horizontal/vertical flips, 90° rotations, brightness jitter
- **Training:** up to 60 epochs with early stopping — stopped at epoch 59, best model at **epoch 47**
- **Test-time augmentation (TTA):** 4-flip TTA was tested on the validation set and adopted because it improved the score (0.6825 vs 0.6798). It is applied only to the final improved model.

## Results

Held-out test set results — baseline model vs the improved model:

| Metric | Baseline | Improved |
|---|---|---|
| Pixel Accuracy | 75.64% | **85.27%** |
| Mean IoU | 0.3785 | **0.5520** |
| Mean Dice | 0.5087 | **0.6841** |
| Weighted F1 | 0.7457 | **0.8475** |

Best validation mIoU: **0.6798** (baseline: 0.3671)

Per-class improvements worth noting:

| Class | Baseline IoU | Improved IoU |
|---|---|---|
| Building-Flooded | 0.33 | **0.66** |
| Road-Flooded | 0.22 | **0.43** |
| Vehicle | 0.07 | **0.26** |

The biggest gain is exactly where it matters — detecting flooded buildings, which doubled compared to the baseline.

## How the flood verdict works

After segmentation, the model adds up the percentage of **Building-Flooded + Road-Flooded** pixels:

- **≥ 1.0%** → Flood-related features detected
- **0.3% – 1.0%**, or low prediction confidence (mean confidence < 0.45) → **Uncertain** — needs human review
- **Water ≥ 8% but flooded infrastructure < 0.3%** → treated as natural water (river/lake), *not* a flood

This rule was added because an early version of the logic treated any large water body as a flood, which is wrong.

## Tech stack

- Python, [PyTorch](https://pytorch.org) (DeepLabV3 implementation)
- [Google Colab](https://colab.research.google.com) for training
- NumPy, Matplotlib for analysis and visualisation
- [Kaggle](https://www.kaggle.com/datasets/imroze/floodnet-cropresized-512-wpartialmasks) for the FloodNet dataset

## Limitations (being honest about them)

- The model is trained only on aerial/UAV images. Ground-level photos (like photos taken from the street) are outside its training data and may give unreliable results.
- The dataset is small (398 images) and covers one event — Hurricane Harvey — so generalisation to other regions is limited.
- Very small classes like Vehicle are still comparatively weak (IoU 0.26).
- The confidence score shown during prediction is pixel-classification confidence, not a probability that a flood occurred.

## Future scope

- Train on a larger, multi-region dataset including actual satellite imagery
- Add a simple web interface where a user uploads an image and gets the verdict instantly
- Combine predictions with map/GPS coordinates so flooded zones can be plotted on a real map
- Try transformer-based segmentation models and compare against DeepLabV3

## References

- FloodNet dataset and benchmark — Rahnemoonfar et al., ["FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding"](https://arxiv.org/abs/2012.02951)
- DeepLabV3 — Chen et al., ["Rethinking Atrous Convolution for Semantic Image Segmentation"](https://arxiv.org/abs/1706.05587)
- MobileNetV3 — Howard et al., ["Searching for MobileNetV3"](https://arxiv.org/abs/1905.02244)

## Author

**K. Saketh Varma**
