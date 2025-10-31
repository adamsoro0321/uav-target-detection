import os
from tqdm import tqdm
from PIL import Image

# chemins
base_dir = "visdrone"
splits = ["train", "val"]

# ID de la classe "car" dans VisDrone (voir doc officielle)
TARGET_CLASS_ID = 4  

for split in splits:
    ann_dir = os.path.join(base_dir, split, "annotations")
    img_dir = os.path.join(base_dir, split, "images")
    out_dir = os.path.join(base_dir, "labels_filtered", split)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Filtrage des annotations pour '{split}'...")
    for file in tqdm(os.listdir(ann_dir)):
        if not file.endswith(".txt"):
            continue
        img_name = file.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path)
        w_img, h_img = img.size

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

                if cls_id != TARGET_CLASS_ID:
                    continue

                x_c = (x + w / 2) / w_img
                y_c = (y + h / 2) / h_img
                w_n = w / w_img
                h_n = h / h_img
                out_lines.append(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

        if out_lines:  # garder seulement les images avec au moins un "car"
            with open(os.path.join(out_dir, file), "w") as out:
                out.write("\n".join(out_lines))
