# 🍒🍅🍓 Fruit Image Classification with CNNs (ResNet‑50)

This project implements an **end‑to‑end deep learning pipeline** for image classification using **PyTorch**.  
The task: correctly identify whether an image contains a **cherry, tomato, or strawberry**.

It demonstrates the complete workflow of modern computer vision:
- **EDA & preprocessing**
- **Baselines** (MLP, custom CNN)
- **Transfer learning** with **ResNet‑50** (final model)

> **Final model:** ResNet‑50 (fine‑tuned)  
> **Final test metrics:**  
> • Test Loss: **0.0775** · Overall Accuracy: **0.9765**  
> • Per‑class Accuracy — Cherry: **0.9510** (466/490), Tomato: **0.9961** (505/507), Strawberry: **0.9818** (485/494)

---

## 🎯 Objectives

- Build a complete **image classification pipeline** (EDA → preprocessing → training → evaluation).  
- Compare architectures: **MLP**, **custom CNN**, **ResNet‑50 transfer learning**.  
- Explore **augmentation**, **optimizers**, **losses**, **regularization**, and **cross‑validation**.  
- Save, load, and evaluate trained models (`.pth`).

---

## 📊 Dataset

- **Total images:** 6,000 (RGB, ~300×300) — **2,000 per class** (cherry, tomato, strawberry).  
- **Split:** 4,500 for training, 1,500 held‑out for testing.  
- **Source:** Images collected from **Flickr** (publicly available).  
- **Included:** Dataset is provided in this repo for reproducibility (respect Flickr terms).

**Normalization stats used (computed from training data):**
```python
mean = [0.5473, 0.4103, 0.3312]
std  = [0.2913, 0.2981, 0.2977]
```

---

## 🏗️ Methodology

### 1) EDA
- Verified **class balance** (2,000 images per class).  
- Checked image sizes, outliers, and sample variability (scale, background, lighting).  
- Computed dataset **mean/std** for normalization.

### 2) Preprocessing & Augmentation
- **Resize:** 300×300; **ToTensor**; **Normalize** with dataset stats.  
- **Augmentations:** random horizontal flip, rotation (±20°), random resized crop, color jitter.  
- **Stratified split** into train/validation to maintain class balance.

### 3) Models
- **Baseline MLP:** simple dense network (for non‑conv baseline).  
- **Custom CNN:** 3×(Conv→ReLU→Pool) + FC, with dropout.  
- **ResNet‑50 (final):** pre‑trained on ImageNet; replaced final FC with 3‑class head and fine‑tuned.

### 4) Training Setup
- **Loss:** CrossEntropyLoss (also experimented with **Focal Loss**).  
- **Optimizers:** Adam / SGD (momentum) / RMSprop; weight decay for regularization.  
- **Scheduler:** StepLR for LR decay on longer runs.  
- **Cross‑Validation:** 5‑fold experiments for stability checks.  
- **Saving:** `torch.save(model.state_dict(), "model.pth")`; **Eval mode** used for testing.

### 5) Evaluation
- **Overall accuracy** + **per‑class accuracy**.  
- **Learning curves** (loss/accuracy vs epochs) for train/val.  
- Sanity checks for data leakage/overfit; confusion trends reviewed qualitatively.

---

## 📈 Results

**Final Model:** ResNet‑50 (fine‑tuned)  
**Final Test Metrics:**
```
Test Loss: 0.0775, Overall Test Accuracy: 0.9765
Accuracy of cherry: 0.9510 (466/490)
Accuracy of tomato: 0.9961 (505/507)
Accuracy of strawberry: 0.9818 (485/494)
```

**Observations:**
- Loss steadily decreased with **augmentation + transfer learning**.  
- The **custom CNN** tended to overfit without strong augmentation; **ResNet‑50** generalized better.

> 📷 **Screenshots:**
> 
1. Sample dataset after augmentation
<p align="center">
  <img src="docs/augment.png" width="750">
</p>

2. Training vs Validation Loss for Resnet-50
<p align="center">
  <img src="docs/loss.png" width="500">
</p>

3. Training vs Validation Accuracy for Resnet-50
<p align="center">
  <img src="docs/accuracy.png" width="500">
</p>


```

---

## 📂 Repository Structure

```text
.
├── train.ipynb          # Training experiments (MLP, CNN, ResNet‑50)
├── test.py              # Script to evaluate saved model on images in testdata/
├── model.pth            # Final trained ResNet‑50 weights (Git LFS)
├── traindata/           # Training dataset (Flickr‑sourced)
├── testdata/            # Example test images
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

> ℹ️ **Large files** are tracked via **Git LFS**. Ensure `git lfs install` before cloning/pulling.

---

## ⚡ How to Run

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Train (Jupyter)
```bash
jupyter notebook train.ipynb
```

### 3) Test (Saved model)
```bash
python test.py
```
This loads `model.pth`, evaluates on images in `./testdata/`, and prints overall + per‑class accuracy.

---

## 🎯 Where This Can Be Used

- **Agriculture:** crop/fruit identification, ripeness/quality inspection, disease screening.  
- **Retail:** automated produce checkout, inventory classification.  
- **Mobile Apps:** consumer education and field recognition tools.  
- **Education/Research:** CNNs vs transfer learning benchmarks and teaching demos.

---

## 🧩 Design Highlights

- Clear baseline → advanced progression (**MLP → CNN → ResNet‑50**).  
- Robust generalization via **augmentation**, **weight decay**, and **transfer learning**.  
- Reproducible results with **saved weights** and scripted evaluation.  
- Practical normalization derived from real training data stats.

---

## 📜 Attribution & Notes

- **Dataset:** Images collected from **Flickr**; resized to 300×300.  
  - If you reuse beyond this project, please **respect Flickr’s licensing/attribution requirements**.  
- **Model:** ResNet‑50 from `torchvision.models` (pre‑trained on ImageNet).  
- **Code:** PyTorch, TorchVision, NumPy, Matplotlib, scikit‑learn utilities.

---


