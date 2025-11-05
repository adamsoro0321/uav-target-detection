import os
import shutil
from tqdm import tqdm
from PIL import Image

base_dir = "visdrone"
splits = ["train", "val"]

# ID original dans VisDrone pour "car"
VISDRONE_CAR_ID = 4
# ID cible dans les labels YOLO filtrés (on met 0 car on n'a qu'une classe)
TARGET_YOLO_ID = 0


def parse_visdrone_line_csv(line):
    """Parse une ligne VisDrone CSV et retourne (x, y, w, h, cls_id) ou None."""
    parts = [p.strip() for p in line.strip().split(",") if p.strip() != ""]
    if len(parts) < 8:
        return None
    try:
        x = float(parts[0])
        y = float(parts[1])
        w = float(parts[2])
        h = float(parts[3])
        cls_id = int(parts[5])  # visdrone: 5th index = class id
        return x, y, w, h, cls_id
    except Exception:
        return None


def parse_yolo_line(line, img_w, img_h):
    """Parse une ligne YOLO (cls x y w h) et retourne (x,y,w,h, cls_id) en pixels."""
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        cls_id = int(float(parts[0]))
        x_c = float(parts[1])
        y_c = float(parts[2])
        w_n = float(parts[3])
        h_n = float(parts[4])
        # convertir en pixel coords si besoin
        x = (x_c - w_n / 2) * img_w
        y = (y_c - h_n / 2) * img_h
        w = w_n * img_w
        h = h_n * img_h
        return x, y, w, h, cls_id
    except Exception:
        return None


for split in splits:
    ann_dir = os.path.join(base_dir, "labels", split)
    img_dir = os.path.join(base_dir, "images", split)

    out_img_dir = os.path.join(base_dir, "car", "images", split)
    out_lbl_dir = os.path.join(base_dir, "car", "labels", split)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    total_files = 0
    copied_images = 0
    written_labels = 0

    print(f"Processing split: {split}")
    for fname in tqdm(os.listdir(ann_dir)):
        if not fname.endswith(".txt"):
            continue
        total_files += 1
        ann_path = os.path.join(ann_dir, fname)
        img_name = fname.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            # si l'image n'existe pas on saute
            continue

        try:
            img = Image.open(img_path)
            w_img, h_img = img.size
        except Exception:
            continue

        # lecture du fichier d'annotation
        with open(ann_path, "r") as f:
            lines = [l for l in f.readlines() if l.strip() != ""]

        out_lines_yolo = []

        for line in lines:
            # tenter CSV (VisDrone) d'abord
            parsed = parse_visdrone_line_csv(line)
            if parsed:
                x, y, w, h, cls_id = parsed
                if cls_id == VISDRONE_CAR_ID:
                    # convertir en format YOLO normalisé (center x,y, width, height)
                    xc = (x + w / 2) / w_img
                    yc = (y + h / 2) / h_img
                    wn = w / w_img
                    hn = h / h_img
                    out_lines_yolo.append(
                        f"{TARGET_YOLO_ID} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"
                    )
                continue

            # sinon tenter format YOLO (cls x y w h déjà normalisé)
            parsed2 = parse_yolo_line(line, w_img, h_img)
            if parsed2:
                x, y, w, h, cls_id = parsed2
                # si le label YOLO avait cls_id correspondant à car après conversion (4->3),
                # vérifier les deux possibilités: soit l'input a été converti (car_id = VISDRONE_CAR_ID - 1)
                # soit l'input garde les IDs originaux. On gère les deux:
                if cls_id == (VISDRONE_CAR_ID - 1) or cls_id == VISDRONE_CAR_ID:
                    # re-normaliser en cas d'arrondi et écrire classe 0
                    xc = (x + w / 2) / w_img
                    yc = (y + h / 2) / h_img
                    wn = w / w_img
                    hn = h / h_img
                    out_lines_yolo.append(
                        f"{TARGET_YOLO_ID} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"
                    )
                continue

            # ligne non reconnue => on skip
            continue

        # si on a au moins une voiture, écrire le label et copier l'image
        if out_lines_yolo:
            label_out_path = os.path.join(out_lbl_dir, fname)
            with open(label_out_path, "w") as out_f:
                out_f.write("\n".join(out_lines_yolo))
            written_labels += 1

            # copy image
            dst_img = os.path.join(out_img_dir, img_name)
            shutil.copy(img_path, dst_img)
            copied_images += 1

    print(
        f"Split {split}: files scanned={total_files}, labels_written={written_labels}, images_copied={copied_images}"
    )
