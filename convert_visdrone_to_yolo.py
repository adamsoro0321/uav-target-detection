import os
from tqdm import tqdm
from PIL import Image

# chemins
base_dir = "dataset/visdrone"
splits = ["train", "val"]

# classes visdrone (0 ignoré)
classes = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}

for split in splits:
    ann_dir = os.path.join(base_dir, split, "annotations")
    img_dir = os.path.join(base_dir, split, "images")
    out_dir = os.path.join(base_dir, "labels", split)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Converting {split} annotations...")
    for file in tqdm(os.listdir(ann_dir)):
        if not file.endswith(".txt"):
            continue
        img_name = file.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path)
            w_img, h_img = img.size
        except:
            continue

        out_lines = []
        with open(os.path.join(ann_dir, file), "r") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",") if p.strip() != ""]
                if len(parts) < 8:
                    continue
                try:
                    x, y, w, h, score, cls_id, trunc, occ = map(int, parts[:8])
                except ValueError:
                    continue

                if cls_id == 0 or cls_id not in classes:
                    continue

                # normalisation YOLO
                x_c = (x + w / 2) / w_img
                y_c = (y + h / 2) / h_img
                w_n = w / w_img
                h_n = h / h_img
                out_lines.append(
                    f"{cls_id - 1} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"
                )

        with open(os.path.join(out_dir, file), "w") as out:
            out.write("\n".join(out_lines))
