#!/usr/bin/env python3
"""Render the official 15-by-5 CV AUC and balanced-accuracy result tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


QUANTITIES = ("25", "50", "100", "200", "all")
FAMILY_ORDER = ("S", "D", "R", "P")


def combination_order(name: str) -> tuple[int, tuple[int, ...]]:
    parts = name.split("+")
    return len(parts), tuple(FAMILY_ORDER.index(part) for part in parts)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
             else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
             else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def colour(value: float) -> tuple[int, int, int]:
    if not np.isfinite(value):
        return 225, 225, 225
    position = float(np.clip((value - 0.4) / 0.5, 0.0, 1.0))
    if position < 0.5:
        ratio = position * 2
        return int(220 + 30 * ratio), int(90 + 150 * ratio), int(85 - 20 * ratio)
    ratio = (position - 0.5) * 2
    return int(250 - 170 * ratio), int(240 - 70 * ratio), int(65 + 40 * ratio)


def centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str,
             selected_font: ImageFont.ImageFont, fill: str = "black") -> None:
    bounds = draw.textbbox((0, 0), text, font=selected_font)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    x = box[0] + (box[2] - box[0] - width) / 2
    y = box[1] + (box[3] - box[1] - height) / 2 - bounds[1]
    draw.text((x, y), text, font=selected_font, fill=fill)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    origin_x: int,
    origin_y: int,
    table: pd.DataFrame,
    combinations: list[str],
    metric: str,
    title: str,
    leader: str,
) -> None:
    label_width, cell_width, cell_height = 175, 82, 43
    header_font, cell_font, label_font = font(19, True), font(17), font(16)
    draw.text((origin_x, origin_y), title, font=font(25, True), fill="#15233b")
    top = origin_y + 45
    for column, quantity in enumerate(QUANTITIES):
        box = (origin_x + label_width + column * cell_width, top,
               origin_x + label_width + (column + 1) * cell_width, top + cell_height)
        draw.rectangle(box, fill="#253b5b", outline="white")
        centered(draw, box, quantity, header_font, "white")
    for row_index, combination in enumerate(combinations):
        y = top + (row_index + 1) * cell_height
        label_box = (origin_x, y, origin_x + label_width, y + cell_height)
        draw.rectangle(label_box, fill="#e8edf4" if combination != leader else "#d8e7ff", outline="white")
        centered(draw, label_box, combination + ("  ★" if combination == leader else ""), label_font)
        for column, quantity in enumerate(QUANTITIES):
            match = table[(table.combination == combination) & (table.quantity.astype(str) == quantity)]
            value = float(match.iloc[0][metric]) if len(match) else float("nan")
            box = (origin_x + label_width + column * cell_width, y,
                   origin_x + label_width + (column + 1) * cell_width, y + cell_height)
            draw.rectangle(box, fill=colour(value), outline="white")
            centered(draw, box, f"{value:.3f}" if np.isfinite(value) else "NA", cell_font)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-summary", type=Path, required=True)
    parser.add_argument("--frozen-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", choices=("composite", "auc", "ba"), default="composite")
    args = parser.parse_args()
    frozen = json.loads(args.frozen_selection.read_text(encoding="utf-8"))
    primary = frozen["primary_feature_set"]
    leader = frozen["leader"]["combination"]
    table = pd.read_csv(args.cv_summary, low_memory=False)
    table = table[(table.feature_set == primary) & (table.training_scope == "all_development")].copy()
    table["auc_fixed_mean"] = table.selection_score
    table["ba_fixed_mean"] = 0.5 * (
        table.human_holdout_balanced_accuracy + table.generator_holdout_balanced_accuracy
    )
    combinations = sorted(table.combination.unique(), key=combination_order)
    if len(combinations) != 15:
        raise ValueError(f"Official table requires all 15 combinations; got {len(combinations)}")
    missing_quantities = set(QUANTITIES) - set(table.quantity.astype(str))
    if missing_quantities:
        raise ValueError(f"Official table is missing quantities: {sorted(missing_quantities)}")

    width = 1320 if args.panel == "composite" else 665
    image = Image.new("RGB", (width, 815), "white")
    draw = ImageDraw.Draw(image)
    if args.panel in {"composite", "auc"}:
        draw_panel(draw, 35, 25, table, combinations, "auc_fixed_mean",
                   "AUC: fixed human/generator source-transfer mean", leader)
    if args.panel in {"composite", "ba"}:
        draw_panel(draw, 690 if args.panel == "composite" else 35, 25, table, combinations,
                   "ba_fixed_mean", "Balanced accuracy: fixed source-transfer mean", leader)
    footer = (f"Primary={primary}  |  ★ frozen CV leader  |  ridge=10, threshold=0.5  |  "
              "locked results never select" if args.panel == "composite" else
              f"Primary={primary}  |  ★ frozen CV leader  |  locked never selects")
    draw.text(
        (35, 770),
        footer,
        font=font(17), fill="#34445d",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)


if __name__ == "__main__":
    main()
