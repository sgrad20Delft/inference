from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import tempfile

def evaluate_detection_accuracy_yolo(prediction_logs, annotation_file):
    """
    Format: prediction_logs[filename] = list of dicts with keys:
    'bbox': [x1, y1, x2, y2], 'score': float, 'class_id': int
    """

    coco_gt = COCO(annotation_file)
    results = []
    normalized_logs = {Path(k).name: v for k, v in prediction_logs.items()}
    images=coco_gt.getImgIds()
    print(f"Images: {len(images)}")
    print(f"Normalized logs: {normalized_logs}")
    for items in normalized_logs.items():
        if items[1]!= []:
            print(items)
    for image_id in images:
        img_info = coco_gt.loadImgs(image_id)[0]
        filename = Path(img_info["file_name"]).name
        # print(f"Image Info:{img_info}")
        if filename not in normalized_logs:
            continue

        detections = normalized_logs[filename]

        if isinstance(detections, dict):
            detections = [detections]
        elif not isinstance(detections, list):
            print(f"[WARNING] Unexpected format for {filename}: {type(detections)}")
            continue

        for det in detections:
            if not isinstance(det, dict) or "bbox" not in det:
                continue
            x1, y1, x2, y2 = det["bbox"]

            # Fix if coordinates are reversed
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1, x2)
            y_max = max(y1, y2)

            w = x_max - x_min
            h = y_max - y_min

            if w <= 0 or h <= 0:
                print(f"[WARNING] Skipping invalid bbox after correction for image_id={image_id}")
                continue

            coco_box = [x_min, y_min, w, h]

            results.append({
                "image_id": image_id,
                "bbox": coco_box,
                "score": det["score"],
                "category_id": det["class_id"]
            })
    print(f"Results: {results}")
    if not results:
        print("[WARNING] No detections found to evaluate.")
        return 0.0, 0

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
        json.dump(results, temp_file)
        temp_file.flush()
        coco_dt = coco_gt.loadRes(temp_file.name)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = coco_eval.stats[0]
    return mAP, len(results)


def evaluate_accuracy(model_perf, task_type, dataset_path, labels_dict=None, index_map=None,
                      annotations_file=None, annotations_path=None, gt_mask_dir=None, gt_panoptic_file=None,
                      predictions_file=None, references_file=None, ground_truth_file=None):
    pred_dict = model_perf.predictions_log

    if task_type == "classification":
        correct = 0
        total = 0
        for filepath, true_label in labels_dict.items():
            filename = Path(filepath).name
            pred_label = pred_dict.get(filename)
            if index_map:
                if str(true_label) in index_map:
                    true_label = index_map[str(true_label)]
                else:
                    print(f"[WARN] Skipping label {true_label} not in class_index_map")
                    continue
            if pred_label == true_label:
                correct += 1
            total += 1
        accuracy = correct / total if total > 0 else 0.0
        print(f"[INFO] Evaluated {total} samples | Correct: {correct} | Accuracy: {accuracy:.4f}")
        return accuracy, total

    elif task_type == "detection":
        if not annotations_file:
            raise ValueError("Missing --annotations_file for detection accuracy")
        mAP, total = evaluate_detection_accuracy_yolo(pred_dict, annotations_path)
        print(f"[INFO] Detection mAP: {mAP:.4f}")
        return mAP, total

    else:
        raise NotImplementedError(f"[ERROR] Accuracy evaluation for task '{task_type}' not yet implemented.")
