import csv
import json
import os
import sys
from pathlib import Path

from datasets import tqdm
from pycocotools.coco import COCO

base = Path(__file__).resolve().parent
base_dir = base.parents[3]

image_dir = os.path.join(base_dir, "dataset_dir", "coco", "coco_images")
label_dir = os.path.join(base_dir, "dataset_dir", "coco", "labels")
output_label_csv = os.path.join(label_dir, "labels_coco.csv")
output_label_json = os.path.join(label_dir, "labels_coco.json")

os.makedirs(label_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

def get_image_ids(annotation_path=None):
    ann_path = annotation_path or os.path.join(base_dir, "dataset_dir", "coco", "annotations", "instances_val2017.json")
    coco = COCO(ann_path)
    return coco.getImgIds()

def get_labels_dict(annotation_path=None):
    ann_path = annotation_path or os.path.join(base_dir, "dataset_dir", "coco", "annotations", "instances_val2017.json")
    coco = COCO(ann_path)
    image_ids = coco.getImgIds()

    all_data = {}
    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        image_annotations = []
        for ann in annotations:
            label = coco.loadCats(ann["category_id"])[0]["name"]
            bbox = ann["bbox"]
            image_annotations.append({"label": label, "bbox": bbox})
        all_data[file_name] = image_annotations

    return all_data

def write_csv_and_json(annotation_path=None):
    ann_path = annotation_path or os.path.join(base_dir, "dataset_dir", "coco", "annotations", "instances_val2017.json")
    coco = COCO(ann_path)
    image_ids = coco.getImgIds()
    print(f"[INFO] Total images: {len(image_ids)}")

    # Write CSV
    with open(output_label_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["file_name", "label", "x", "y", "width", "height"])

        for img_id in tqdm(image_ids, desc="Saving labels to CSV"):
            img_info = coco.loadImgs(img_id)[0]
            file_name = img_info["file_name"]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            annotations = coco.loadAnns(ann_ids)

            for ann in annotations:
                label = coco.loadCats(ann["category_id"])[0]["name"]
                x, y, w, h = ann["bbox"]
                writer.writerow([file_name, label, x, y, w, h])

    # Write JSON
    all_data = get_labels_dict(annotation_path=ann_path)
    with open(output_label_json, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"COCO labels saved to: {output_label_csv} and {output_label_json}")


if __name__ == "__main__":
    annotation_path = sys.argv[1] if len(sys.argv) > 1 else None
    write_csv_and_json(annotation_path)
