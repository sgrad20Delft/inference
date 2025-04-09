import torch
import numpy as np

def postprocess_onnx_output(output, task_type, conf_thresh=0.25, iou_thresh=0.45):
    """
    Postprocess YOLOv4 ONNX output.

    Args:
        output: raw model output (np.array of shape [1, N, 85])
        task_type: e.g., "detection"
        conf_thresh: confidence threshold for filtering
        iou_thresh: IoU threshold for NMS

    Returns:
        list of dicts: [{bbox, score, class_id}]
    """
    if task_type != "detection":
        return output

    raw_output = output[0]  # shape: (1, 52, 52, 3, 85)
    print(f"[DEBUG] Raw output shape: {raw_output.shape}")

    # Flatten detections to (N, 85)
    detections = raw_output.reshape(-1, 85)
    print(f"[INFO] Detections shape: {detections.shape}")
    if detections.shape[1] != 85:
        raise ValueError(
            f"[ERROR] Unexpected output shape: expected 85 features per detection, got {detections.shape[1]}")
    # shape: (num_boxes, 85)
    boxes = []
    for det in detections:
        x, y, w, h = det[:4]
        objectness = det[4]
        class_scores = det[5:]
        class_id = np.argmax(class_scores)
        score = objectness * class_scores[class_id]

        if score > conf_thresh:
            # Convert box to x1, y1, x2, y2
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            boxes.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(score),
                "class_id": int(class_id)
            })

    # Optionally, apply NMS here
    return boxes
