import cv2
import numpy as np
from tqdm import tqdm
import os

DATASET_PATH = "metal_nut"
OUTPUT_PATH = "yolo_metal_nut"

DEFECT_CLASS_MAP = {
    "bent": 0,
    "color": 1,
    "flip": 2,
    "scratch": 3,
}

DEFAULT_CLASS = 0


def create_dirs():
    for split in ["train", "val"]:
        os.makedirs(f"{OUTPUT_PATH}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_PATH}/labels/{split}", exist_ok=True)


def mask_to_bboxes(mask):
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary)

    bboxes = []
    for label_id in range(1, num_labels):
        coords = np.column_stack(np.where(labels == label_id))
        if coords.size == 0:
            continue

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        bboxes.append((x_min, y_min, x_max, y_max))

    return bboxes


def bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h):
    x_center = (x_min + x_max) / 2 / img_w
    y_center = (y_min + y_max) / 2 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return x_center, y_center, width, height


def safe_imread(path, flag=cv2.IMREAD_COLOR):
    img = cv2.imread(path, flag)
    if img is None:
        print(f"[WARN] could not read: {path}")
    return img


def write_empty_labels(path):
    with open(path, "w"):
        pass


# 🔥 TRAIN
def convert_train():
    test_dir = os.path.join(DATASET_PATH, "test")
    gt_dir = os.path.join(DATASET_PATH, "ground_truth")

    for main_type in os.listdir(test_dir):
        main_path = os.path.join(test_dir, main_type)

        if not os.path.isdir(main_path):
            continue

        # GOOD images
        if main_type == "good":
            files = os.listdir(main_path)
            split_idx = int(0.8 * len(files))
            train_files = files[:split_idx]

            print(f"\n[train/good] {len(train_files)} images")

            for img_name in tqdm(train_files, desc="train/good"):
                img_path = os.path.join(main_path, img_name)
                img = safe_imread(img_path)
                if img is None:
                    continue

                stem = os.path.splitext(img_name)[0]
                out_stem = f"good_{stem}"

                cv2.imwrite(f"{OUTPUT_PATH}/images/train/{out_stem}.png", img)
                write_empty_labels(f"{OUTPUT_PATH}/labels/train/{out_stem}.txt")

        # DEFECT images
        elif main_type == "defect":
            for defect_type in os.listdir(main_path):
                img_folder = os.path.join(main_path, defect_type)

                if not os.path.isdir(img_folder):
                    continue

                files = os.listdir(img_folder)
                split_idx = int(0.8 * len(files))
                train_files = files[:split_idx]

                print(f"\n[train/{defect_type}] {len(train_files)} images")

                class_id = DEFECT_CLASS_MAP.get(defect_type, DEFAULT_CLASS)

                for img_name in tqdm(train_files, desc=f"train/{defect_type}"):
                    img_path = os.path.join(img_folder, img_name)
                    img = safe_imread(img_path)
                    if img is None:
                        continue

                    stem = os.path.splitext(img_name)[0]
                    out_stem = f"{defect_type}_{stem}"

                    cv2.imwrite(f"{OUTPUT_PATH}/images/train/{out_stem}.png", img)

                    labels_path = f"{OUTPUT_PATH}/labels/train/{out_stem}.txt"

                    # correct mask path
                    mask_path = os.path.join(
                        gt_dir, defect_type, f"{stem}_mask.png"
                    )

                    mask = safe_imread(mask_path, cv2.IMREAD_GRAYSCALE)

                    if mask is None:
                        print(f"[ERROR] Mask not found: {mask_path}")
                        write_empty_labels(labels_path)
                        continue

                    bboxes = mask_to_bboxes(mask)

                    if not bboxes:
                        write_empty_labels(labels_path)
                        continue

                    h, w = mask.shape
                    lines = []

                    for (x_min, y_min, x_max, y_max) in bboxes:
                        xc, yc, bw, bh = bbox_to_yolo(
                            x_min, y_min, x_max, y_max, w, h
                        )
                        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

                    with open(labels_path, "w") as f:
                        f.write("\n".join(lines))


# 🔥 VAL
def convert_val():
    test_dir = os.path.join(DATASET_PATH, "test")
    gt_dir = os.path.join(DATASET_PATH, "ground_truth")

    for main_type in os.listdir(test_dir):
        main_path = os.path.join(test_dir, main_type)

        if not os.path.isdir(main_path):
            continue

        # GOOD
        if main_type == "good":
            files = os.listdir(main_path)
            split_idx = int(0.8 * len(files))
            val_files = files[split_idx:]

            print(f"\n[val/good] {len(val_files)} images")

            for img_name in tqdm(val_files, desc="val/good"):
                img = safe_imread(os.path.join(main_path, img_name))
                if img is None:
                    continue

                stem = os.path.splitext(img_name)[0]
                out_stem = f"good_{stem}"

                cv2.imwrite(f"{OUTPUT_PATH}/images/val/{out_stem}.png", img)
                write_empty_labels(f"{OUTPUT_PATH}/labels/val/{out_stem}.txt")

        # DEFECT
        elif main_type == "defect":
            for defect_type in os.listdir(main_path):
                img_folder = os.path.join(main_path, defect_type)

                if not os.path.isdir(img_folder):
                    continue

                files = os.listdir(img_folder)
                split_idx = int(0.8 * len(files))
                val_files = files[split_idx:]

                print(f"\n[val/{defect_type}] {len(val_files)} images")

                class_id = DEFECT_CLASS_MAP.get(defect_type, DEFAULT_CLASS)

                for img_name in tqdm(val_files, desc=f"val/{defect_type}"):
                    img = safe_imread(os.path.join(img_folder, img_name))
                    if img is None:
                        continue

                    stem = os.path.splitext(img_name)[0]
                    out_stem = f"{defect_type}_{stem}"

                    cv2.imwrite(f"{OUTPUT_PATH}/images/val/{out_stem}.png", img)

                    labels_path = f"{OUTPUT_PATH}/labels/val/{out_stem}.txt"

                    mask_path = os.path.join(
                        gt_dir, defect_type, f"{stem}_mask.png"
                    )

                    mask = safe_imread(mask_path, cv2.IMREAD_GRAYSCALE)

                    if mask is None:
                        print(f"[ERROR] Mask not found: {mask_path}")
                        write_empty_labels(labels_path)
                        continue

                    bboxes = mask_to_bboxes(mask)

                    if not bboxes:
                        write_empty_labels(labels_path)
                        continue

                    h, w = mask.shape
                    lines = []

                    for (x_min, y_min, x_max, y_max) in bboxes:
                        xc, yc, bw, bh = bbox_to_yolo(
                            x_min, y_min, x_max, y_max, w, h
                        )
                        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

                    with open(labels_path, "w") as f:
                        f.write("\n".join(lines))


def write_yaml():
    names = {v: k for k, v in DEFECT_CLASS_MAP.items()}
    nc = len(names)

    names_str = "\n".join(f"  {i}: {names[i]}" for i in range(nc))

    yaml_content = f"""path: {os.path.abspath(OUTPUT_PATH)}
train: images/train
val: images/val
nc: {nc}
names:
{names_str}
"""

    yaml_path = f"{OUTPUT_PATH}/dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n[✓] dataset.yaml written: {yaml_path}")


if __name__ == "__main__":
    create_dirs()
    convert_train()
    convert_val()
    write_yaml()
    print("\n[✓] Conversion completed")