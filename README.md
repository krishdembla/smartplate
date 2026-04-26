# SmartPlate — Vision-Based Healthy Plate Scoring

**CSCI B457 Computer Vision · Spring 2026 · Final Project · Group 9**  
Krish Dembla · Riyan Patel · Andrew Berman

---

SmartPlate converts a single photograph of a food plate into a pixel-level food segmentation, a 7-group nutritional breakdown, and a 0–100 health score based on the Harvard Healthy Eating Plate guidelines. The full pipeline runs on consumer hardware (Apple Silicon MPS or CUDA).

---

## What is included

| Item | Description |
|------|-------------|
| `README.md` | This file |
| `B457_Report__Smartplate` | Project report (CVPR 2026 format) |
| `code/` | All source code, trained model checkpoints, data assets, and sample output figures |

No raw dataset files are included — see the Dataset section below.

---


## Dataset

**FoodSeg103** (Wu et al., ACM MM 2021) — 7,118 food images with dense pixel-level masks across 103 ingredient categories. The dataset is not bundled in this submission. On first run, notebook `01` will automatically download it (~500 MB) via HuggingFace into `~/.cache/huggingface/`. No manual steps are needed.

HuggingFace dataset ID: `EduardoPacheco/FoodSeg103`

The two files in `assets/` are hand-built lookup tables specific to this project and are not part of the original FoodSeg103 dataset.

---

## Setup

```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open any notebook in Jupyter Lab or VS Code and run from top to bottom.

---

## Running the notebooks

Trained model weights are already included in `checkpoints/`, so `04_demo.ipynb` can be run directly without retraining anything.

| Notebook | Purpose |
|----------|---------|
| `01_explore_and_prepare.ipynb` | Downloads FoodSeg103 and explores class/group distributions |
| `02_train_segformer.ipynb` | Fine-tunes SegFormer-B0 on FoodSeg103 (~40 min on Colab T4) |
| `03_train_baseline.ipynb` | Trains the ResNet-50 multi-label classifier and reports per-group F1 |
| `04_demo.ipynb` | Loads the SegFormer checkpoint and runs the full pipeline on validation and custom images |

---

## Results summary

| Metric | Value |
|--------|-------|
| mIoU — 103 food classes | 14.7% |
| mIoU — 7 food groups | 51.3% |
| ResNet-50 baseline mean F1 | 78.4% |

Top individual class IoU: broccoli 80.1%, carrot 71.7%, corn 68.9%.  
Best food groups by IoU: vegetables 67.8%, whole grains 66.1%.

---

## Health scoring

The 0–100 score is computed in `smartplate.score_plate` using three components:

- **Base score**: fraction of food-area pixels assigned to positive groups (vegetables, fruits, whole grains, healthy protein)
- **Balance penalty**: L1 distance from Harvard target proportions, reduces the base score by up to 15% when positive groups are imbalanced
- **Negative penalty**: weighted deduction for refined grains, red/processed meat, and sugary/fatty foods

A plate that exactly matches Harvard Healthy Eating Plate targets scores 100. A plate consisting entirely of unhealthy items scores near 0.

The 7 food groups and their role in scoring:

| Group | Role |
|-------|------|
| Vegetables | Positive (target: 30%) |
| Fruits | Positive (target: 20%) |
| Whole Grains | Positive (target: 25%) |
| Healthy Protein | Positive (target: 25%) |
| Refined Grains | Negative (penalty weight: 0.15) |
| Red/Processed Meat | Negative (penalty weight: 0.35) |
| Sugary/Fatty | Negative (penalty weight: 0.80) |
