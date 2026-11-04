"""
Builds two figures for the ISIC-2018 classification dataset:
1. A horizontal bar chart showing the number of images per diagnostic class.
2. A 2x4 grid of representative images, one per class, sampled from the held-out test split.
The figures are saved to the inference_outputs/dataset_figures directory.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


CLASS_INFO = {
    "MEL": ("Melanoma", "Malignant", "#b83232"),
    "BCC": ("Basal Cell Carcinoma", "Malignant", "#d1534f"),
    "AKIEC": ("Actinic Keratosis", "Malignant", "#e87868"),
    "NV": ("Melanocytic Nevus", "Benign", "#2368a2"),
    "BKL": ("Benign Keratosis", "Benign", "#3d83b8"),
    "DF": ("Dermatofibroma", "Benign", "#62a0c6"),
    "VASC": ("Vascular Lesion", "Benign", "#86bbd8"),
}
CLASS_ORDER = ["MEL", "BCC", "AKIEC", "NV", "BKL", "DF", "VASC"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("datasets/ISIC-2018/classification"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("inference_outputs/dataset_figures"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_dataset_table(dataset_dir):
    tables = []
    for split in ("train", "valid", "test"):
        labels_path = dataset_dir / split / "labels.csv"
        images_dir = dataset_dir / split / "images"
        if not labels_path.exists():
            raise FileNotFoundError(labels_path)
        table = pd.read_csv(labels_path)
        table["split"] = split
        table["image_path"] = table["image"].map(lambda name: images_dir / f"{name}.jpg")
        tables.append(table)

    dataset = pd.concat(tables, ignore_index=True)
    if dataset["image"].duplicated().any():
        duplicates = dataset.loc[dataset["image"].duplicated(), "image"].tolist()
        raise ValueError(f"Images appear in more than one split: {duplicates[:5]}")
    dataset["class"] = dataset[CLASS_ORDER].idxmax(axis=1)
    return dataset


def square_image(path, size=640):
    with Image.open(path) as source:
        image = source.convert("RGB")
        return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)


def build_image_grid(dataset, output_dir, seed):
    test = dataset[dataset["split"] == "test"]
    rng = np.random.default_rng(seed)
    selected = []
    for class_code in CLASS_ORDER:
        candidates = test[(test["class"] == class_code) & test["image_path"].map(Path.exists)]
        if candidates.empty:
            candidates = dataset[(dataset["class"] == class_code) & dataset["image_path"].map(Path.exists)]
        selected.append(candidates.iloc[int(rng.integers(len(candidates)))])

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.2), facecolor="white")
    axes = axes.ravel()
    for axis, row in zip(axes, selected):
        class_code = row["class"]
        class_name, category, color = CLASS_INFO[class_code]
        axis.imshow(square_image(row["image_path"]))
        axis.set_title(class_name, fontsize=12, fontweight="semibold", pad=8)
        axis.text(
            .5, -.055, f"{class_code}  •  {category}  •  {row['image']}",
            transform=axis.transAxes, ha="center", va="top", fontsize=8.5, color=color,
        )
        axis.axis("off")

    axes[-1].axis("off")
    fig.suptitle("Representative images from the seven diagnostic categories",
                 fontsize=18, fontweight="bold", y=.985)
    fig.text(.5, .018, "One reproducibly sampled image per class from the held-out test split.",
             ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=.035, right=.965, top=.91, bottom=.085, wspace=.12, hspace=.28)
    path = output_dir / "seven_category_image_grid.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path, selected


def build_distribution_chart(dataset, output_dir):
    counts = dataset["class"].value_counts().reindex(CLASS_ORDER).astype(int)
    total = int(counts.sum())
    labels = [CLASS_INFO[code][0] for code in CLASS_ORDER]
    colors = [CLASS_INFO[code][2] for code in CLASS_ORDER]

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axis = plt.subplots(figsize=(11.5, 6.2), facecolor="white")
    positions = np.arange(len(CLASS_ORDER))
    bars = axis.barh(positions, counts.values, color=colors, height=.66)
    axis.set_yticks(positions, [f"{name} ({code})" for name, code in zip(labels, CLASS_ORDER)])
    axis.invert_yaxis()
    axis.set_xlabel("Number of images")
    axis.set_title(f"Diagnostic-class distribution (n = {total:,})",
                   fontsize=17, fontweight="bold", pad=15)
    axis.xaxis.grid(True, color="#e4e7eb", linewidth=.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    maximum = counts.max()
    axis.set_xlim(0, maximum * 1.18)
    for bar, count in zip(bars, counts.values):
        percentage = 100 * count / total
        axis.text(count + maximum * .012, bar.get_y() + bar.get_height() / 2,
                  f"{count:,}  ({percentage:.1f}%)", va="center", fontsize=10)

    fig.text(.5, .02,
             "Malignant classes are shown in red; benign classes are shown in blue. "
             "Counts combine the local train, validation, and test splits.",
             ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=.25, right=.96, top=.88, bottom=.14)
    path = output_dir / "class_distribution_bar_chart.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)

    counts_table = pd.DataFrame({
        "class_code": CLASS_ORDER,
        "class_name": labels,
        "category": [CLASS_INFO[code][1] for code in CLASS_ORDER],
        "count": counts.values,
        "percentage": 100 * counts.values / total,
    })
    counts_table.to_csv(output_dir / "class_distribution.csv", index=False)
    return path, counts_table


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset_table(args.dataset_dir)
    grid_path, selected = build_image_grid(dataset, args.output_dir, args.seed)
    chart_path, counts = build_distribution_chart(dataset, args.output_dir)

    print(counts.to_string(index=False, formatters={"percentage": "{:.2f}%".format}))
    print("\nRepresentative images")
    for row in selected:
        print(f"  {row['class']}: {row['image']} ({row['split']})")
    print(f"\nSaved image grid: {grid_path}")
    print(f"Saved distribution chart: {chart_path}")


if __name__ == "__main__":
    main()