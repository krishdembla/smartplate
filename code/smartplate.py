"""SmartPlate shared helpers.

Pure functions for:
    - Aggregating predicted segmentation masks into food-group proportions.
    - Scoring a plate against the Harvard Healthy Eating Plate rubric.
    - Looking up nutritional macros per food group.
    - Visualizing masks as overlays on the original image.
    - Running a single-image SegFormer inference.

All functions take plain Python/NumPy inputs so they can be unit-tested
without a model loaded.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image


# ---- Food groups and Harvard Healthy Eating Plate rubric --------------------

POSITIVE_GROUPS: Tuple[str, ...] = (
    "vegetables",
    "fruits",
    "whole_grains",
    "healthy_protein",
)

NEGATIVE_GROUPS: Tuple[str, ...] = (
    "refined_grains",
    "red_processed_meat",
    "sugary_fatty",
)

ALL_GROUPS: Tuple[str, ...] = POSITIVE_GROUPS + NEGATIVE_GROUPS

# Harvard Healthy Eating Plate target proportions (fractions of the food area).
HARVARD_TARGETS: Dict[str, float] = {
    "vegetables": 0.30,
    "fruits": 0.20,
    "whole_grains": 0.25,
    "healthy_protein": 0.25,
}

# Relative importance of each positive group when penalizing shortfalls.
# Weights sum to 1 across the positive groups.
POSITIVE_WEIGHTS: Dict[str, float] = {
    "vegetables": 0.30,
    "fruits": 0.20,
    "whole_grains": 0.25,
    "healthy_protein": 0.25,
}

# Penalty strength for each negative group's proportion on the plate.
NEGATIVE_PENALTIES: Dict[str, float] = {
    "refined_grains": 0.15,      # rice/bread are suboptimal, not harmful
    "red_processed_meat": 0.35,  # penalized but has protein value
    "sugary_fatty": 0.80,        # junk food stays heavily penalized
}

# Red meat counts as partial protein toward the positive base (0 = none, 1 = full).
RED_MEAT_PROTEIN_CREDIT: float = 0.50


# ---- Class-to-group mapping loading -----------------------------------------

def load_class_to_group(csv_path: str | Path) -> Dict[int, str]:
    """Load a hand-built class_id → food_group mapping from CSV.

    CSV columns: class_id,class_name,food_group
    Rows with an empty food_group or food_group == "background" are skipped.
    """
    mapping: Dict[int, str] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = (row.get("food_group") or "").strip()
            if not group or group == "background":
                continue
            if group not in ALL_GROUPS:
                raise ValueError(
                    f"Unknown food_group '{group}' for class {row['class_id']}. "
                    f"Must be one of {ALL_GROUPS} or 'background'."
                )
            mapping[int(row["class_id"])] = group
    return mapping


# ---- Mask → proportions -----------------------------------------------------

def mask_to_proportions(
    mask: np.ndarray,
    class_to_group: Dict[int, str],
    background_class: int = 0,
) -> Dict[str, float]:
    """Aggregate a per-pixel class-id mask into per-food-group proportions.

    Pixels whose class is not in ``class_to_group`` (including background) are
    treated as non-food and excluded from the denominator, so proportions sum
    to 1 across the food groups (or are all zero if the plate is empty).
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got shape {mask.shape}")

    group_counts = {g: 0 for g in ALL_GROUPS}
    total_food = 0
    unique, counts = np.unique(mask, return_counts=True)
    for class_id, count in zip(unique.tolist(), counts.tolist()):
        if class_id == background_class:
            continue
        group = class_to_group.get(int(class_id))
        if group is None:
            continue
        group_counts[group] += int(count)
        total_food += int(count)

    if total_food == 0:
        return {g: 0.0 for g in ALL_GROUPS}
    return {g: group_counts[g] / total_food for g in ALL_GROUPS}


# ---- Scoring ---------------------------------------------------------------

def score_plate(proportions: Dict[str, float]) -> float:
    """0–100 healthy-plate score from food-group proportions.

    Decomposes into three signals:
      base       = 100 × (positive food fraction + half-credit for red meat protein)
      balance    = up to 15% reduction of base for imbalance vs Harvard targets
      penalty    = deductions proportional to negative-group coverage

    Ideal Harvard plate → 100. Empty/all-junk plate → 0. Results clipped to [0, 100].
    """
    for g in ALL_GROUPS:
        if g not in proportions:
            raise KeyError(f"Missing proportion for group '{g}'")

    if sum(proportions.values()) == 0:
        return 0.0

    # Red meat contributes partial credit as a protein source toward the base.
    protein_credit = proportions["red_processed_meat"] * RED_MEAT_PROTEIN_CREDIT
    positive_frac = sum(proportions[g] for g in POSITIVE_GROUPS) + protein_credit
    base = 100.0 * positive_frac

    # L1 distance to ideal targets, normalized to [0, 1] (max L1 is 2).
    l1 = sum(abs(proportions[g] - HARVARD_TARGETS[g]) for g in POSITIVE_GROUPS)
    balance_penalty = base * 0.15 * (l1 / 2.0)

    negative_penalty = 100.0 * sum(
        NEGATIVE_PENALTIES[g] * proportions[g] for g in NEGATIVE_GROUPS
    )

    score = base - balance_penalty - negative_penalty
    return max(0.0, min(100.0, score))


# ---- Nutritional lookup -----------------------------------------------------

def load_macros(csv_path: str | Path) -> Dict[str, Dict[str, float]]:
    """Load per-food-group macros from CSV.

    CSV columns: food_group,kcal_per_100g,protein_g,carb_g,fat_g
    """
    table: Dict[str, Dict[str, float]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            table[row["food_group"].strip()] = {
                "kcal": float(row["kcal_per_100g"]),
                "protein_g": float(row["protein_g"]),
                "carb_g": float(row["carb_g"]),
                "fat_g": float(row["fat_g"]),
            }
    return table


def proportions_to_macros(
    proportions: Dict[str, float],
    macros_table: Dict[str, Dict[str, float]],
    total_grams: float = 400.0,
) -> Dict[str, float]:
    """Estimate total kcal / protein / carb / fat from food-group proportions.

    ``total_grams`` is a fixed assumption about plate mass (no depth info);
    the report should acknowledge this as a rough estimate.
    """
    totals = {"kcal": 0.0, "protein_g": 0.0, "carb_g": 0.0, "fat_g": 0.0}
    for group, frac in proportions.items():
        if group not in macros_table:
            continue
        grams = frac * total_grams
        per_g = {k: v / 100.0 for k, v in macros_table[group].items()}
        for k in totals:
            totals[k] += grams * per_g[k]
    return totals


# ---- Visualization ----------------------------------------------------------

def _distinct_colors(n: int, seed: int = 0) -> np.ndarray:
    """Return ``n`` visually distinct RGB colors (uint8)."""
    rng = np.random.default_rng(seed)
    hsv = np.stack(
        [
            np.linspace(0, 1, n, endpoint=False),
            rng.uniform(0.55, 0.9, size=n),
            rng.uniform(0.7, 1.0, size=n),
        ],
        axis=1,
    )
    # HSV → RGB (vectorized, same math as colorsys.hsv_to_rgb)
    h, s, v = hsv[:, 0] * 6.0, hsv[:, 1], hsv[:, 2]
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    rgb = np.zeros_like(hsv)
    mask = i == 0; rgb[mask] = np.stack([v[mask], t[mask], p[mask]], 1)
    mask = i == 1; rgb[mask] = np.stack([q[mask], v[mask], p[mask]], 1)
    mask = i == 2; rgb[mask] = np.stack([p[mask], v[mask], t[mask]], 1)
    mask = i == 3; rgb[mask] = np.stack([p[mask], q[mask], v[mask]], 1)
    mask = i == 4; rgb[mask] = np.stack([t[mask], p[mask], v[mask]], 1)
    mask = i == 5; rgb[mask] = np.stack([v[mask], p[mask], q[mask]], 1)
    return (rgb * 255).astype(np.uint8)


def overlay_mask(
    image: Image.Image | np.ndarray,
    mask: np.ndarray,
    num_classes: int = 104,
    alpha: float = 0.55,
    background_class: int = 0,
) -> np.ndarray:
    """Blend a class-id mask over an image. Returns RGB uint8 array."""
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = image
    if image_np.shape[:2] != mask.shape:
        raise ValueError(
            f"image {image_np.shape[:2]} and mask {mask.shape} shapes differ"
        )

    palette = _distinct_colors(num_classes)
    color_mask = palette[mask]
    bg = mask == background_class
    blended = image_np.copy().astype(np.float32)
    blended[~bg] = (
        (1 - alpha) * blended[~bg] + alpha * color_mask[~bg].astype(np.float32)
    )
    return blended.clip(0, 255).astype(np.uint8)


# ---- Inference --------------------------------------------------------------

def predict_mask(model, processor, image: Image.Image, device: str = "mps") -> np.ndarray:
    """Run a HuggingFace SegFormer on a single PIL image. Returns H×W class-id mask."""
    import torch

    inputs = processor(images=image, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, C, h/4, w/4)
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],  # (H, W)
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.int32)


# ---- Quick self-tests -------------------------------------------------------

def _self_tests() -> None:
    """Sanity checks that can run without a trained model."""
    ideal = {
        "vegetables": 0.30, "fruits": 0.20,
        "whole_grains": 0.25, "healthy_protein": 0.25,
        "refined_grains": 0.0, "red_processed_meat": 0.0, "sugary_fatty": 0.0,
    }
    assert abs(score_plate(ideal) - 100.0) < 1e-6, "Ideal plate should score 100"

    all_sugar = {g: 0.0 for g in ALL_GROUPS}
    all_sugar["sugary_fatty"] = 1.0
    assert score_plate(all_sugar) <= 5.0, f"All-sugar should score near 0, got {score_plate(all_sugar)}"

    veg_only = {g: 0.0 for g in ALL_GROUPS}
    veg_only["vegetables"] = 1.0
    s = score_plate(veg_only)
    assert 70 < s < 100, f"All-vegetable plate should score high, got {s}"

    steak_only = {g: 0.0 for g in ALL_GROUPS}
    steak_only["red_processed_meat"] = 1.0
    s = score_plate(steak_only)
    assert 5 < s < 25, f"All-steak plate should score low but not 0, got {s}"

    empty = {g: 0.0 for g in ALL_GROUPS}
    assert score_plate(empty) == 0.0

    print("smartplate self-tests passed.")


if __name__ == "__main__":
    _self_tests()
