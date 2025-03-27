import os
import numpy as np
import glob
import torch
import json
import pycocotools.coco as coco
import pycocotools.cocoeval as cocoeval
from PIL import Image
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ---------------------------------
# 1. Classification Accuracy
# ---------------------------------
def evaluate_classification_accuracy(model_loader,preprocessor,dataset_dir,labels_dict):
    """
        Evaluate classification accuracy over a dataset.
        labels_dict should map image filename to ground truth label.
        Returns (accuracy, sample_count).
        """
    image_files=glob.glob(os.path.join(dataset_dir,"*"))
    total=0
    correct=0
    for img_path in image_files:
        filename=os.path.basename(img_path)
        if filename not in labels_dict:
            processed_input=preprocessor.preprocess(img_path)
            output=model_loader.infer(processed_input)
            if isinstance(output,torch.Tensor):
                pred=output.argmax().item()
            elif isinstance(output,list):
                pred=int(np.argmax(output[0]))
            else:
                raise ValueError("Unsupported output type from inference.")
            if pred==labels_dict[filename]:
                correct+=1
            total+=1
    accuracy=correct/total if total>0 else 0
    return accuracy,total

# ---------------------------------
# 2. Object Detection mAP (BBox)
# ---------------------------------


def evaluate_detection_accuracy(model_loader,preprocessor,dataset_dir,labels_dict):
    """
        Evaluate object detection performance by computing mAP for bounding boxes.
        Ground truth annotations must be in COCO JSON format.
        The model should output a list of detections per image with keys: 'bbox', 'score', 'category_id'.
        Returns (mAP, number_of_images).
        """
    coco_gt=coco.COCO(labels_dict)
    results=[]
    image_ids=coco_gt.getImgIds()
    for img_id in image_ids:
        img_filename=coco_gt.loadImgs(img_id)[0]['file_name']
        img_path=os.path.join(dataset_dir,img_filename)
        processed_input=preprocessor.preprocess(img_path)
        output=model_loader.infer(processed_input)
        if isinstance(output,list):
            for det in output:
                det['image_id']=img_id
                results.append(det)
        else:
            raise ValueError("Detection model output must be a list")
    coco_dt=coco_gt.loadRes(results)
    coco_eval=cocoeval.COCOeval(coco_gt,coco_dt,'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAP=coco_eval.stats[0]
    return mAP,len(image_ids)

# ---------------------------------
# 3. Instance Segmentation mAP
# ---------------------------------
def evaluate_segmentation_accuracy(model_loader,preprocessor,dataset_dir,labels_dict):
    """
        Evaluate instance segmentation performance using mAP.
        Ground truth annotations must be in COCO JSON format.
        The model should output predictions including a 'segmentation' field.
        Returns (mAP, number_of_images).
        """
    coco_gt=coco.COCO(labels_dict)
    results=[]
    image_ids=coco_gt.getImgIds()
    for img_id in image_ids:
        img_info=coco_gt.loadImgs(img_id)[0]
        img_path=os.path.join(dataset_dir,img_info['file_name'])
        processed_input=preprocessor.preprocess(img_path)
        output=model_loader.infer(processed_input)
        if isinstance(output,list):
            for det in output:
                det['image_id']=img_id
                results.append(det)
        else:
            raise ValueError("Instance segmentation model output must be a list.")

    coco_dt=coco_gt.loadRes(results)
    coco_eval=cocoeval.COCOeval(coco_gt,coco_dt,'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAP=coco_eval.stats[0]
    return mAP,len(image_ids)

# ---------------------------------
# 4. Semantic Segmentation mIoU
# ---------------------------------
def compute_iou(gt_mask,pred_mask,num_classes):
    """
        Compute Intersection over Union (IoU) for each class.
        Returns a list of IoU values for classes present in ground truth.
        """
    ious=[]
    for cls in range(num_classes):
        gt_cls=(gt_mask==cls)
        pred_cls=(pred_mask==cls)
        intersection=np.logical_and(gt_cls,pred_cls).sum()
        union=np.logical_or(gt_cls,pred_cls).sum()
        iou=intersection/union if union>0 else 0
        if union==0:
            continue
        ious.append(intersection/union)
    return ious

def evaluate_segmentation_miou(model_loader, preprocessor, dataset_dir, gt_mask_dir, num_classes):
    """
        Evaluate semantic segmentation by computing mean IoU (mIoU).
        For each image in dataset_dir, a corresponding ground truth mask is expected in gt_mask_dir.
        Returns (mIoU, number_of_images).
        """
    image_files=glob.glob(os.path.join(dataset_dir,"*"))
    total_iou=[]
    valid_images=0
    for img_path in image_files:
        filename=os.path.basename(img_path)
        gt_mask_path=os.path.join(gt_mask_dir,filename)
        if not os.path.exists(gt_mask_path):
            continue
        gt_mask=np.array(Image.open(gt_mask_path))
        processed_input=preprocessor.preprocess(img_path)
        output=model_loader.infer(processed_input)
        if isinstance(output,torch.Tensor):
            pred_mask=output.squeeze(0).argmax(dim=0).cpu().numpy()
        elif isinstance(output,np.ndarray):
            if output.ndim==4:
                pred_mask=np.argmax(output,axis=1).squeeze(0)
            else:
                pred_mask=np.argmax(output,axis=1)
        else:
            raise ValueError("Unsupported segmentation model output type.")
        ious=compute_iou(gt_mask,pred_mask,num_classes)
        if ious:
            total_iou.append(np.mean(ious))
            valid_images+=1
    mIoU=np.mean(total_iou) if valid_images>0 else 0
    return mIoU,valid_images
# ---------------------------------
# 5. Panoptic Segmentation PQ (Panoptic Quality)
# ---------------------------------
def evaluate_panoptic_quality(model_loader, preprocessor, dataset_dir, gt_panoptic_file):
    """
        Evaluate panoptic segmentation performance using Panoptic Quality (PQ).
        Ground truth annotations must be in COCO panoptic JSON format.
        Note: A full implementation requires the panopticapi package.
        Here we provide a placeholder function.
        """
    try:
        from panopticapi.evaluation import pq_compute, prepare_for_panoptic_eval
    except ImportError:
        raise ImportError("panopticapi is required for panoptic segmentation evaluation.")

    # Placeholder: In a full implementation, run the model on each image, save predictions in the required format,
    # then call prepare_for_panoptic_eval and pq_compute.
    pq=0.0
    return pq,None
# ---------------------------------
# 6. Keypoint Detection mAP
# ---------------------------------
def evaluate_keypoint_detection_accuracy(model_loader,preprocessor,dataset_dir,gt_annotations_file):
    """
        Evaluate keypoint detection performance using mAP.
        Ground truth annotations must be in COCO JSON format.
        The model should output a list of keypoint detections per image.
        Returns (AP for keypoints, number_of_images).
        """
    coco_gt=coco.COCO(gt_annotations_file)
    results=[]
    image_ids=coco_gt.getImgIds()
    for img_id in image_ids:
        img_info=coco_gt.loadImgs(img_id)[0]
        img_path=os.path.join(dataset_dir,img_info['file_name'])
        processed_input=preprocessor.preprocess(img_path)
        output=model_loader.infer(processed_input)
        if isinstance(output,list):
            for det in output:
                det['image_id']=img_id
                results.append(det)
        else:
            raise ValueError("Keypoint detection model output must be a list.")
    coco_dt=coco_gt.loadRes(results)
    coco_eval=cocoeval.COCOeval(coco_gt,coco_dt,'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    ap_keypoints=coco_eval.stats[0]
    return ap_keypoints,len(image_ids)

# ---------------------------------
# 7. Image Captioning BLEU Score
# ---------------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def evaluate_image_captioning_bleu(predictions_file,references_file):
    """
        Evaluate image captioning performance using BLEU score.
        predictions_file: JSON file with predictions in the format:
            [{"image_id": id, "caption": "predicted caption"}, ...]
        references_file: JSON file with ground truth captions in COCO Captions format:
            {"annotations": [{"image_id": id, "caption": "reference caption"}, ...]}
        Returns (average BLEU score, number_of_evaluated_images).
        """
    with open (predictions_file,'r') as f:
        predictions=json.load(f)
    with open(references_file,'r') as f:
        refs=json.load(f)
    ref_dict={}
    for ann in refs.get('annotations',[]):
        image_id=ann['image_id']
        caption=ann['caption']
        if image_id not in ref_dict:
            ref_dict[image_id]=[]
        ref_dict[image_id].append(caption.split())
    smoothing_function=SmoothingFunction().method1
    total_bleu=0
    count=0
    for pred in predictions:
        image_id=pred['image_id']
        caption_tokens=pred['caption'].split()
        if image_id not in ref_dict:
            continue
        bleu=sentence_bleu([ref_dict[image_id]],caption_tokens,smoothing_function=smoothing_function)
        total_bleu+=bleu
        count+=1
    avg_bleu=total_bleu/count if count>0 else 0
    return avg_bleu,count

# ---------------------------------
# 8. Image Retrieval mAP
# ---------------------------------
def compute_cosine_similarity(vec1,vec2):
    if np.linalg.norm(vec1)==0 or np.linalg.norm(vec2)==0:
        return 0.0
    return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
def compute_average_precision(ranked_list, relevant_set):
    """
    Compute Average Precision (AP) for a single query.
    ranked_list: list of gallery image IDs ranked in descending order.
    relevant_set: set of relevant gallery image IDs.
    """
    if not relevant_set:
        return 0.0
    hit_count=0
    precision_sum=0.0
    for i,gallery_id in enumerate(ranked_list):
        if gallery_id in relevant_set:
            hit_count+=1
            precision=hit_count/(i+1)
            precision_sum+=precision
    return precision_sum/len(relevant_set)
def evaluate_image_retrieval_accuracy(model_loader,preprocessor,query_dir,gallery_dir,ground_truth_file,metric='cosine',top_k=10):
    """
        Evaluate image retrieval performance using mean Average Precision (mAP).

        Args:
          model_loader: A model loader that outputs embedding vectors.
          preprocessor: Preprocessing module.
          query_dir: Directory containing query images.
          gallery_dir: Directory containing gallery images.
          ground_truth_file: JSON file mapping query image filenames to a list of relevant gallery filenames.
          metric: 'cosine' (default) or 'euclidean' for similarity measure.
          top_k: Number of top gallery items to consider.

        Returns:
          (mAP, number_of_queries)
        """
    with open(ground_truth_file,'r') as f:
        gt_mapping=json.load(f)
    gallery_files=glob.glob(os.path.join(gallery_dir,"*"))
    gallery_embeddings={}
    for file_path in gallery_files:
        gallery_id=os.path.basename(file_path)
        processed_input=preprocessor.preprocess(file_path)
        output=model_loader.infer(processed_input)
        if isinstance(output,torch.Tensor):
            embedding=output.squeeze(0).cpu().numpy()
        elif isinstance(output,np.ndarray):
            embedding=output.squeeze(0)
        else:
            raise ValueError("Unsupported output type for image retrieval embedding.")
        gallery_embeddings[gallery_id]=embedding
    query_files=sorted(glob.glob(os.path.join(query_dir,"*")))
    ap_list=[]
    num_queries=0

    for file_path in query_files:
        query_id=os.path.basename(file_path)
        if query_id not in gt_mapping:
            continue
        processed_input=preprocessor.preprocess(file_path)
        output=model_loader.infer(processed_input)
        if isinstance(output,torch.Tensor):
            query_emb=output.squeeze(0).cpu().numpy()
        elif isinstance(output,np.ndarray):
            query_emb=output.squeeze(0)
        else:
            raise ValueError("Unsupported output type for image retrieval embedding.")
        scores={}
        for gallery_id,embedding in gallery_embeddings.items():
            if metric=='cosine':
                sim=compute_cosine_similarity(query_emb,embedding)
            elif metric=='euclidean':
                sim=-np.linalg.norm(query_emb-embedding)
            else:
                raise ValueError(f"Unknown metric. Choose 'cosine' or 'euclidean'.")
            scores[gallery_id]=sim
        ranked_gallery=sorted(scores,key=lambda x:scores[x],reverse=True)[:top_k]
        relevant_set=set(gt_mapping[query_id])
        ap=compute_average_precision(ranked_gallery,relevant_set)
        ap_list.append(ap)
        num_queries+=1
    mAP=np.mean(ap_list) if num_queries>0 else 0.0
    return mAP,num_queries
