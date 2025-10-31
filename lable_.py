import os
import shutil
from tqdm import tqdm

# Dossiers source
base_dir = "visdrone"
splits = ["train", "val"]

# Classe "car" correspond à ID = 4 dans VisDrone → devient 3 après (cls_id - 1)
car_class_id = 4 - 1

for split in splits:
    ann_dir = os.path.join(base_dir, "labels", split)
    img_dir = os.path.join(base_dir, "images", split)

    # Dossiers de sortie
    out_img_dir = os.path.join(base_dir, "car", "images", split)
    out_lbl_dir = os.path.join(base_dir, "car", "labels", split)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    print(f"Processing {split} split...")
    for file in tqdm(os.listdir(ann_dir)):
        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(ann_dir, file)
        with open(file_path, "r") as f:
            lines = f.readlines()

        # On filtre uniquement les lignes contenant la classe 'car'
        car_lines = [l for l in lines if l.strip().startswith(str(car_class_id))]

        # Si le fichier contient au moins une voiture
        if car_lines:
            # Copier le label filtré
            with open(os.path.join(out_lbl_dir, file), "w") as out_f:
                out_f.writelines(car_lines)

            # Copier l’image correspondante
            img_name = file.replace(".txt", ".jpg")
            src_img = os.path.join(img_dir, img_name)
            dst_img = os.path.join(out_img_dir, img_name)

            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
