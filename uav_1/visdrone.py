from __future__ import annotations

from pathlib import Path
import shutil

from ultralytics.utils.downloads import download
from ultralytics.utils import ASSETS_URL, TQDM


def visdrone2yolo(
    base_dir: str | Path, split: str, source_name: str | None = None
) -> None:
    """Convert VisDrone annotations to YOLO format.

    Args:
        base_dir: Base directory path for the dataset
        split: Dataset split ('train', 'val', or 'test')
        source_name: Source folder name (defaults to VisDrone2019-DET-{split})
    """
    from PIL import Image

    # Use Path objects for proper path handling
    dir_path = Path(base_dir)
    source_dir = dir_path / (source_name or f"VisDrone2019-DET-{split}")
    images_dir = dir_path / "images" / split
    labels_dir = dir_path / "labels" / split
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Move images to new structure
    source_images_dir = source_dir / "images"
    if source_images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        for img_path in source_images_dir.glob("*.jpg"):
            shutil.move(str(img_path), str(images_dir / img_path.name))

    # Process annotations
    annotations_dir = source_dir / "annotations"
    if not annotations_dir.exists():
        print(f"Warning: Annotations directory not found: {annotations_dir}")
        return

    for annotation_file in TQDM(
        annotations_dir.glob("*.txt"), desc=f"Converting {split}"
    ):
        img_file = images_dir / annotation_file.with_suffix(".jpg").name
        if not img_file.exists():
            continue

        try:
            with Image.open(img_file) as img:
                img_size = img.size
        except Exception as e:
            print(f"Error opening image {img_file}: {e}")
            continue

        dw, dh = 1.0 / img_size[0], 1.0 / img_size[1]
        lines: list[str] = []

        with open(annotation_file, encoding="utf-8") as file:
            for row in [x.split(",") for x in file.read().strip().splitlines()]:
                if len(row) >= 6 and row[4] != "0":  # Skip ignored regions
                    try:
                        x, y, w, h = map(int, row[:4])
                        cls = int(row[5]) - 1

                        # Validate class ID (0-9 for VisDrone)
                        if not 0 <= cls <= 9:
                            continue

                        # Convert to YOLO format
                        x_center = (x + w / 2) * dw
                        y_center = (y + h / 2) * dh
                        w_norm = w * dw
                        h_norm = h * dh

                        lines.append(
                            f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                        )
                    except (ValueError, IndexError):
                        continue  # Skip malformed lines

        if lines:
            (labels_dir / annotation_file.name).write_text(
                "".join(lines), encoding="utf-8"
            )


def main() -> None:
    """Main function to download and convert VisDrone dataset."""
    # Download (ignores test-challenge split)
    base_dir = Path("./dataset/visdrone")

    # Convert
    splits = {
        "VisDrone2019-DET-train": "train",
        "VisDrone2019-DET-val": "val",
        "VisDrone2019-DET-test-dev": "test",
    }

    for folder, split in splits.items():
        print(f" folder  {folder} split {split} split...")
        visdrone2yolo(base_dir, split, folder)
        # Keep original folders - do not delete them
        # folder_path = base_dir / folder
        # if folder_path.exists():
        #     shutil.rmtree(folder_path)


if __name__ == "__main__":
    main()
