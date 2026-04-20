# SmartPlate — Vision-Based Healthy Plate Scoring

CSCI B457 Computer Vision · Spring 2026 · Final Project · Group 9

---

SmartPlate takes a single photo of a food plate and produces a semantic segmentation mask, a 0–100 health score, and estimated macros. The pipeline is: image → SegFormer-B0 (fine-tuned on FoodSeg103) → per-pixel food class labels → aggregate to 7 Harvard food groups → score against Harvard Healthy Eating Plate targets.

---

## What is included in this submission

- **`README.md`** — this file.
- **`code/`** — all runnable code: the shared `smartplate.py` module, four Jupyter notebooks, trained model checkpoints, hand-built data assets, and output figures.
- **`report/Group_09.pdf`** — the project report (CVPR 2026 format).
- **No raw dataset files.** FoodSeg103 is not bundled — it auto-downloads on first notebook run (see Dataset section below).

---

## Repository layout

```
├── README.md
├── code/
│   ├── smartplate.py                      # scoring, nutrition, visualization, inference
│   ├── notebooks/
│   │   ├── 01_explore_and_prepare.ipynb   # dataset EDA, scaffolds class→group CSV
│   │   ├── 02_train_segformer.ipynb       # SegFormer-B0 fine-tuning (ran on Colab T4)
│   │   ├── 03_train_baseline.ipynb        # ResNet50 multi-label baseline
│   │   └── 04_demo.ipynb                  # end-to-end demo: image → mask → score
│   ├── assets/
│   │   ├── foodseg103_to_groups.csv       # hand-built mapping: 104 classes → 7 food groups
│   │   └── usda_macros.csv                # per-group macro reference (kcal/protein/carb/fat per 100g)
│   ├── checkpoints/
│   │   ├── segformer_best/                # trained SegFormer-B0 weights (safetensors)
│   │   └── resnet50_multilabel.pt         # trained ResNet50 baseline weights
│   ├── figures/                           # output PNGs referenced in the report
│   └── requirements.txt
└── report/
    └── Group_09.pdf                       # final report
```

---

## Dataset

**FoodSeg103** (Wu et al., ACM MM 2021) — 7,120 images across 103 food classes plus background. The dataset is not included in this zip. On first run, notebook `01` (or any notebook that calls `load_dataset`) will automatically download it (~500 MB) from HuggingFace into `~/.cache/huggingface/`. No manual download or setup is needed.

HuggingFace mirror: https://huggingface.co/datasets/EduardoPacheco/FoodSeg103  
Original paper: https://arxiv.org/abs/2105.05409

The two small data assets that *are* included (`assets/foodseg103_to_groups.csv` and `assets/usda_macros.csv`) are hand-built lookup tables used for group aggregation and macro estimation — they are not part of the original dataset.

---

## Setup

Tested on macOS with Apple Silicon (PyTorch MPS). Also runs on Linux/Colab with CUDA.

```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name smartplate --display-name "SmartPlate (.venv)"
```

Open any notebook in Jupyter or VS Code and select the **SmartPlate (.venv)** kernel.

---

## Running the pipeline

The trained checkpoints are included, so `04_demo.ipynb` can be run directly without retraining. Notebooks 02 and 03 are only needed to reproduce training from scratch.

1. `01_explore_and_prepare.ipynb` — downloads FoodSeg103 and visualizes samples. The class→group CSV is already filled so this is mainly for reference.
2. `02_train_segformer.ipynb` — fine-tunes SegFormer-B0 and saves the checkpoint. Full training ran on Colab T4 (~40 min).
3. `03_train_baseline.ipynb` — trains the ResNet50 multi-label classifier and reports per-group F1.
4. `04_demo.ipynb` — loads the SegFormer checkpoint and runs the full scoring pipeline on held-out plates and custom photos.

---

## Trained model results (validation set)

| Metric | Value |
|--------|-------|
| mIoU — 103 food classes | 14.7% |
| mIoU — 7 food groups | 51.3% |
| ResNet50 baseline mean F1 | 78.4% |

Top segmentation classes: broccoli 80.1%, carrot 71.7%, corn 68.9%.
Best food groups: vegetables 67.8%, whole grains 66.1%.

---

## Scoring

The health score is defined in `smartplate.score_plate`. It decomposes into a base score (fraction of the plate covered by positive food groups), a balance penalty (L1 distance from Harvard Healthy Eating Plate targets), and a negative penalty (weighted proportion of refined grains, red/processed meat, and sugary/fatty foods). An ideal Harvard plate scores 100; a plate of junk food scores near 0.

The 7 food groups are: vegetables, fruits, whole grains, healthy protein (positive); refined grains, red/processed meat, sugary/fatty (negative).

---

## Dependencies

- HuggingFace `transformers` — `SegformerForSemanticSegmentation` (pretrained on ADE20K).
- `torchvision.models.resnet50` (IMAGENET1K_V2 weights).
- Full citations in `report/main.bib`.
