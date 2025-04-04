import os
from pathlib import Path

import numpy as np
import glob
import torch
import json
import pycocotools.coco as coco
import pycocotools.cocoeval as cocoeval
from PIL import Image
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import os
from math import ceil

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ---------------------------------------------------------------------------
# 1. Classification Accuracy
# ---------------------------------------------------------------------------
def evaluate_classification_accuracy(model_perf, dataset_path,limit=None):


    correct = 0
    total = 0

    dataset_path = Path(dataset_path)
    all_paths = list(dataset_path.glob("**/*.png")) + list(dataset_path.glob("**/*.jpg")) + list(dataset_path.glob("**/*.jpeg"))
    sample_paths = all_paths
    if limit:
        sample_paths = sample_paths[:limit]

    print(f"Running accuracy evaluation on {len(sample_paths)} samples...")
    with open("vision/dataset_dir/mnist/labels/labels.json", "r") as f:
        labels_dict = json.load(f)



    batch_size = 50
    num_batches = ceil(len(sample_paths) / batch_size)

    for batch_idx in range(num_batches):
        print(f"Batch {batch_idx + 1}/{num_batches}")
        batch_paths = sample_paths[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_tensors = []
        batch_labels = []

        for sample_path in batch_paths:
            filename = sample_path.name
            # print("Evaluating: ", filename, " ...")
            if filename not in labels_dict:
                print(f"Warning: {filename} not found in labels_dict; skipping.")
                continue
            label = labels_dict[filename]
            image_tensor = model_perf.preprocess_fn.preprocess(image_path=sample_path)
            batch_tensors.append(image_tensor)
            # print("Image tensor shape: ", image_tensor.shape)
            batch_labels.append(label)
            # print("Label: ", label)

        if not batch_tensors:
            continue

        input_tensor = torch.cat(batch_tensors, dim=0)
        predictions = model_perf.predict(input_tensor)
        print("Predictions: ", predictions, " ...")
        for pred, label in zip(predictions, batch_labels):
            print(f"Pred: {pred}, Label: {label}, Match: {pred == label}")
            if pred == label:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Accuracy over {total} samples: {accuracy:.4f}")
    return accuracy, total


# ---------------------------------------------------------------------------
# 2. Object Detection mAP (BBox)
# ---------------------------------------------------------------------------
def evaluate_detection_accuracy(
    model_perf,
    dataset_dir: str,
    labels_dict_json: str
):
    """
    Evaluate object detection performance by computing mAP (COCO bbox metric).

    labels_dict_json is a path to a COCO JSON with bounding box annotations.

    The model should output a list of detection dicts per image, e.g.:
      [
        {"bbox": [x, y, w, h], "score": float, "category_id": int},
        ...
      ]

    We use model_perf for both preprocessor + inference.
    """
    coco_gt = coco.COCO(labels_dict_json)
    results = []
    image_ids = coco_gt.getImgIds()

    for img_id in image_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(dataset_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        processed_input = model_perf.preprocess_fn.preprocess(img_path)
        output = model_perf.model_loader.infer(processed_input)

        if not isinstance(output, list):
            raise ValueError("Detection model output must be a list of dicts.")
        for det in output:
            # Ensure 'bbox', 'score', 'category_id' are present
            det["image_id"] = int(img_id)
            results.append(det)

    # Convert results list to JSON string for loadRes
    results_json = json.dumps(results)
    coco_dt = coco_gt.loadRes(results_json)
    coco_eval = cocoeval.COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAP = coco_eval.stats[0]
    return mAP, len(image_ids)


# ---------------------------------------------------------------------------
# 3. Instance Segmentation mAP
# ---------------------------------------------------------------------------
def evaluate_segmentation_accuracy(
    model_perf,
    dataset_dir: str,
    labels_dict_json: str
):
    """
    Evaluate instance segmentation performance using COCO 'segm' mAP.

    labels_dict_json is the COCO JSON path.
    The model output should be a list of dicts, each with e.g.:
      {"segmentation": ..., "score": float, "category_id": int}

    We use model_perf for both preprocessor + inference.
    """
    coco_gt = coco.COCO(labels_dict_json)
    results = []
    image_ids = coco_gt.getImgIds()

    for img_id in image_ids:
        print("Starting Evaluation for Image ID: ", img_id, " ...")
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(dataset_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        processed_input = model_perf.preprocess_fn.preprocess(img_path)
        output = model_perf.model_loader.infer(processed_input)

        if not isinstance(output, list):
            raise ValueError("Instance segmentation model output must be a list of dicts.")

        for det in output:
            det["image_id"] = int(img_id)
            results.append(det)

    # Convert results list to JSON string for loadRes
    results_json = json.dumps(results)
    coco_dt = coco_gt.loadRes(results_json)
    coco_eval = cocoeval.COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAP = coco_eval.stats[0]
    return mAP, len(image_ids)


# ---------------------------------------------------------------------------
# 4. Semantic Segmentation mIoU
# ---------------------------------------------------------------------------
def compute_iou(gt_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int):
    """
    Compute per-class IoU. Returns a list of IoUs for classes found in ground truth.
    """
    ious = []
    for cls_idx in range(num_classes):
        gt_cls = (gt_mask == cls_idx)
        pred_cls = (pred_mask == cls_idx)
        intersection = np.logical_and(gt_cls, pred_cls).sum()
        union = np.logical_or(gt_cls, pred_cls).sum()
        if union == 0:
            continue
        iou = float(intersection) / float(union)
        ious.append(iou)
    return ious

def evaluate_segmentation_miou(
    model_perf,
    dataset_dir: str,
    gt_mask_dir: str,
    num_classes: int
):
    """
    Evaluate semantic segmentation via mean IoU (mIoU).
    For each image in dataset_dir, there's a corresponding mask in gt_mask_dir with the same filename.
    We use model_perf for preprocessor + inference.
    """
    image_files = glob.glob(os.path.join(dataset_dir, "*"))
    total_iou = []
    valid_images = 0

    for img_path in image_files:
        filename = os.path.basename(img_path)
        gt_mask_path = os.path.join(gt_mask_dir, filename)
        if not os.path.exists(gt_mask_path):
            continue

        gt_mask = np.array(Image.open(gt_mask_path))
        processed_input = model_perf.preprocess_fn.preprocess(img_path)
        output = model_perf.model_loader.infer(processed_input)

        # If model outputs [B, C, H, W], we do argmax on dim=1
        if isinstance(output, torch.Tensor):
            pred_mask = output.squeeze(0).argmax(dim=0).cpu().numpy()
        elif isinstance(output, np.ndarray):
            if output.ndim == 4:
                pred_mask = np.argmax(output, axis=1).squeeze(0)
            else:
                pred_mask = np.argmax(output, axis=1)
        else:
            raise ValueError("Unsupported segmentation output type.")

        ious = compute_iou(gt_mask, pred_mask, num_classes)
        if ious:
            total_iou.append(np.mean(ious))
            valid_images += 1

    mIoU = float(np.mean(total_iou)) if valid_images > 0 else 0.0
    return mIoU, valid_images


# ---------------------------------------------------------------------------
# 5. Panoptic Segmentation PQ
# ---------------------------------------------------------------------------
def evaluate_panoptic_quality(
    model_perf,
    dataset_dir: str,
    gt_panoptic_file: str
):
    """
    Placeholder for Panoptic Quality (PQ). A full implementation requires panopticapi
    plus properly formatted predictions, and calling pq_compute().

    We use model_perf for preprocessor + inference if you incorporate actual logic.
    """
    try:
        from panopticapi.evaluation import pq_compute, prepare_for_panoptic_eval  # type: ignore
    except ImportError:
        raise ImportError("panopticapi is required for panoptic segmentation evaluation.")

    # In a real scenario: run model on each image, build predictions, call pq_compute, etc.
    # example:
    #  processed_input = model_perf.preprocess_fn.preprocess(...)
    #  output = model_perf.model_loader.infer(processed_input)
    pq_val = 0.0
    return pq_val, None


# ---------------------------------------------------------------------------
# 6. Keypoint Detection mAP
# ---------------------------------------------------------------------------
def evaluate_keypoint_detection_accuracy(
    model_perf,
    dataset_dir: str,
    gt_annotations_file: str
):
    """
    Evaluate keypoint detection performance via COCO 'keypoints' mAP.
    The model output is a list of dicts with e.g.:
      {"keypoints": [x1,y1,v1, x2,y2,v2, ...], "score": float, "category_id": int}

    We use model_perf for preprocessor + inference.
    """
    coco_gt = coco.COCO(gt_annotations_file)
    results = []
    image_ids = coco_gt.getImgIds()

    for img_id in image_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(dataset_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        processed_input = model_perf.preprocess_fn.preprocess(img_path)
        output = model_perf.model_loader.infer(processed_input)

        if not isinstance(output, list):
            raise ValueError("Keypoint detection model output must be a list of dicts.")

        for det in output:
            det["image_id"] = int(img_id)
            results.append(det)

    # Convert results list to JSON string for loadRes
    results_json = json.dumps(results)
    coco_dt = coco_gt.loadRes(results_json)
    coco_eval = cocoeval.COCOeval(coco_gt, coco_dt, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    ap_kpts = float(coco_eval.stats[0])
    return ap_kpts, len(image_ids)


# ---------------------------------------------------------------------------
# 7. Image Captioning BLEU Score
# ---------------------------------------------------------------------------
def evaluate_image_captioning_bleu(
    predictions_file: str,
    references_file: str
):
    """
    Evaluate image captioning performance using NLTK's BLEU score.

    predictions_file: JSON, e.g. [{"image_id": 123, "caption": "predicted text"}, ...]
    references_file: JSON with ground truth in COCO-style captions, e.g.:
      {
        "annotations": [
          {"image_id": 123, "caption": "reference text"},
          ...
        ]
      }

    Returns: (avg_bleu, num_evaluated)

    (Uses external files rather than your model_perf. If you want to generate
     predictions on-the-fly, adapt similarly by calling model_perf as needed.)
    """
    with open(predictions_file, "r") as f:
        predictions = json.load(f)
    with open(references_file, "r") as f:
        refs = json.load(f)

    # We'll store references as: ref_dict[image_id] = [ ["tokens",...], ["tokens2",...] ]
    ref_dict = {}
    for ann in refs.get("annotations", []):
        img_id = int(ann["image_id"])
        caption_str = str(ann["caption"])
        if img_id not in ref_dict:
            ref_dict[img_id] = []
        ref_dict[img_id].append(caption_str.split())

    total_bleu = 0.0
    count = 0
    smoothing_function = SmoothingFunction().method1

    for pred in predictions:
        img_id = int(pred["image_id"])
        caption_tokens = str(pred["caption"]).split()

        if img_id not in ref_dict:
            continue

        references = ref_dict[img_id]
        hypothesis = caption_tokens
        bleu = sentence_bleu(
            references,
            hypothesis,
            smoothing_function
        )
        total_bleu += bleu
        count += 1

    avg_bleu = total_bleu / count if count > 0 else 0.0
    return avg_bleu, count


# ---------------------------------------------------------------------------
# 8. Image Retrieval mAP
# ---------------------------------------------------------------------------
def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Return 0 if either vector is zero-length, else the cosine similarity.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def compute_average_precision(ranked_list: list, relevant_set: set) -> float:
    """
    Compute Average Precision (AP) for a single query.

    ranked_list: list of gallery IDs (sorted descending by similarity)
    relevant_set: set of IDs considered ground truth for that query
    """
    if not relevant_set:
        return 0.0

    hit_count = 0
    precision_sum = 0.0
    for i, gallery_id in enumerate(ranked_list):
        if gallery_id in relevant_set:
            hit_count += 1
            precision_sum += (hit_count / (i + 1))

    return precision_sum / len(relevant_set)

def evaluate_image_retrieval_accuracy(
    model_perf,
    query_dir: str,
    gallery_dir: str,
    ground_truth_file: str,
    metric: str = "cosine",
    top_k: int = 10
):
    """
    Evaluate image retrieval performance using mean Average Precision (mAP).

    ground_truth_file: JSON mapping query filenames -> list of relevant gallery filenames.
    metric: 'cosine' or 'euclidean'.
    top_k: limit rank list.

    We use model_perf for preprocessor + inference of query + gallery images.
    """
    with open(ground_truth_file, "r") as f:
        gt_mapping = json.load(f)

    # Build gallery embeddings
    gallery_files = glob.glob(os.path.join(gallery_dir, "*"))
    gallery_embeddings = {}

    for file_path in gallery_files:
        gallery_id = os.path.basename(file_path)
        processed_input = model_perf.preprocess_fn.preprocess(file_path)
        output = model_perf.model_loader.infer(processed_input)

        if isinstance(output, torch.Tensor):
            emb = output.squeeze(0).cpu().numpy()
        elif isinstance(output, np.ndarray):
            emb = output.squeeze(0)
        else:
            raise ValueError("Unsupported output type for retrieval embedding.")

        gallery_embeddings[gallery_id] = emb

    # Evaluate queries
    query_files = sorted(glob.glob(os.path.join(query_dir, "*")))
    ap_list = []
    num_queries = 0

    for file_path in query_files:
        query_id = os.path.basename(file_path)
        if query_id not in gt_mapping:
            continue

        processed_input = model_perf.preprocess_fn.preprocess(file_path)
        output = model_perf.model_loader.infer(processed_input)

        if isinstance(output, torch.Tensor):
            query_emb = output.squeeze(0).cpu().numpy()
        elif isinstance(output, np.ndarray):
            query_emb = output.squeeze(0)
        else:
            raise ValueError("Unsupported output type for retrieval embedding.")

        # Calculate similarity or negative distance
        scores = {}
        for g_id, emb in gallery_embeddings.items():
            if metric == "cosine":
                sim_val = compute_cosine_similarity(query_emb, emb)
            elif metric == "euclidean":
                # negative distance => bigger is better
                sim_val = -float(np.linalg.norm(query_emb - emb))
            else:
                raise ValueError("Unknown metric. Use 'cosine' or 'euclidean'.")

            scores[g_id] = sim_val

        # Rank gallery by descending similarity
        ranked_gallery = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        relevant_set = set(gt_mapping[query_id])
        ap_val = compute_average_precision(ranked_gallery, relevant_set)
        ap_list.append(ap_val)
        num_queries += 1

    mean_ap = float(np.mean(ap_list)) if num_queries > 0 else 0.0
    return mean_ap, num_queries
