import os
import sys
import cv2
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

# Grounding DINO
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# COCO Evaluation
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Add your custom paths
current_dir = os.path.dirname(os.path.abspath(__file__))  
data_prepare_dir = os.path.join(os.path.dirname(current_dir), 'data_prepare')
if data_prepare_dir not in sys.path:
    sys.path.append(data_prepare_dir)

from coco_c import apply_corruption 

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_iou(boxA, boxB):
    """두 Bounding Box 간의 IoU를 계산합니다. (box format: [x, y, w, h])"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

def initialize_text_prototypes(model, processor, categories):
    prototypes = {}
    for cat in categories:
        cat_name = cat['name']
        cat_id = cat['id']
        inputs = processor(text=[cat_name], return_tensors="pt").to(model.device)
        with torch.no_grad():
            text_outputs = model.model.text_backbone(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            cls_features = text_outputs.last_hidden_state[:, 0, :]
            projected_features = model.model.text_projection(cls_features)
            projected_features = F.normalize(projected_features, p=2, dim=-1)
        prototypes[cat_id] = projected_features.squeeze(0).clone()
    return prototypes

def extract_object_queries(outputs, valid_boxes, target_size):
    object_queries = outputs.decoder_hidden_states[-1][0] 
    raw_boxes = outputs.pred_boxes[0] 
    
    W, H = target_size
    extracted_queries = []
    for box in valid_boxes:
        x1, y1, x2, y2 = box.tolist()
        cx = (x1 + x2) / 2.0 / W
        cy = (y1 + y2) / 2.0 / H
        bw = (x2 - x1) / float(W)
        bh = (y2 - y1) / float(H)
        norm_box = torch.tensor([cx, cy, bw, bh], device=raw_boxes.device)
        
        distances = F.l1_loss(raw_boxes, norm_box.unsqueeze(0).expand_as(raw_boxes), reduction='none').sum(dim=1)
        best_idx = torch.argmin(distances)
        
        query = object_queries[best_idx]
        query = F.normalize(query, p=2, dim=-1)
        extracted_queries.append(query)
        
    if len(extracted_queries) == 0:
        return torch.empty(0, object_queries.size(-1)).to(object_queries.device)
    return torch.stack(extracted_queries)

def update_instance_prototypes(query_features, probs, labels, prototypes, alpha=0.01, T=20):
    for i, label_idx in enumerate(labels):
        w = probs[i]
        if w >= 1e-1:
            w_new = 1 - torch.exp(-w / T)
            current_proto = prototypes[label_idx]
            feat = query_features[i]
            
            updated_proto = (1 - w_new) * current_proto + w_new * feat
            refined_proto = alpha * current_proto + (1 - alpha) * updated_proto
            refined_proto = F.normalize(refined_proto, p=2, dim=-1)
            prototypes[label_idx] = refined_proto
    return prototypes

def main():
    set_seed(1)
    
    ann_file = './data/coco/annotations/instances_val2017.json'
    img_dir = './data/coco/val2017'
    res_file = './grounding_dino_predictions.json'
    
    # 🚨 진단 로그 및 이미지를 저장할 폴더 생성
    diag_dir = "./diagnostics"
    os.makedirs(diag_dir, exist_ok=True)
    poisoning_logs = []
    
    print("Loading COCO annotations...")
    cocoGt = COCO(ann_file)
    img_ids = cocoGt.getImgIds()
    
    categories = cocoGt.loadCats(cocoGt.getCatIds())
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    cat_name_to_id = {cat['name']: cat['id'] for cat in categories}
    text_prompt = " . ".join([cat['name'] for cat in categories]) + " ."

    print("Loading Grounding DINO model...")
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, device_map="auto")
    model.eval()
    
    prototypes = initialize_text_prototypes(model, processor, categories)
    results_list = []
    
    print(f"Starting inference on {len(img_ids)} images...")
    for i, img_id in enumerate(tqdm(img_ids, desc="Evaluating with PTA")):
        img_info = cocoGt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            continue
            
        img_bgr = cv2.imread(img_path)
        img_bgr_corrupted = apply_corruption(img_bgr, corruption_name='gaussian_noise', severity=3)
        img_rgb = cv2.cvtColor(img_bgr_corrupted, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        inputs = processor(images=pil_img, text=text_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        target_sizes = [pil_img.size[::-1]]
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.4, text_threshold=0.3, target_sizes=target_sizes
        )[0]
        
        boxes = results["boxes"]
        scores = results["scores"]
        labels_text = results["labels"]
        
        detected_cat_ids = []
        valid_indices = []
        for idx, label_text in enumerate(labels_text):
            clean_label = label_text.strip().lower()
            if clean_label in cat_name_to_id:
                detected_cat_ids.append(cat_name_to_id[clean_label])
                valid_indices.append(idx)
                
                x1, y1, x2, y2 = boxes[idx].tolist()
                w, h = x2 - x1, y2 - y1
                results_list.append({
                    "image_id": img_id,
                    "category_id": cat_name_to_id[clean_label],
                    "bbox": [x1, y1, w, h],
                    "score": scores[idx].item()
                })
        
        if len(valid_indices) > 0:
            valid_boxes = boxes[valid_indices]
            valid_scores = scores[valid_indices]
            
            ann_ids = cocoGt.getAnnIds(imgIds=img_id)
            gt_anns = cocoGt.loadAnns(ann_ids)
            
            query_features = extract_object_queries(
                outputs=outputs, valid_boxes=valid_boxes, target_size=pil_img.size
            )
            
            original_prototypes = {k: v.clone() for k, v in prototypes.items()}
            
            prototypes = update_instance_prototypes(
                query_features=query_features, probs=valid_scores, 
                labels=detected_cat_ids, prototypes=prototypes, alpha=0.01, T=20
            )

            # 🚨 [저장 로직 추가] 오염 이벤트 발생 시 데이터 기록 및 이미지 저장
            for idx, pred_cat_id in enumerate(detected_cat_ids):
                pred_box = [valid_boxes[idx][0].item(), valid_boxes[idx][1].item(), 
                            (valid_boxes[idx][2]-valid_boxes[idx][0]).item(), 
                            (valid_boxes[idx][3]-valid_boxes[idx][1]).item()]
                
                max_iou = 0.0
                for gt in gt_anns:
                    if gt['category_id'] == pred_cat_id:
                        iou = calculate_iou(pred_box, gt['bbox'])
                        if iou > max_iou: max_iou = iou
                
                # 독성 업데이트 조건 (IoU < 0.3 이면서 Confidence > 0.4)
                if max_iou < 0.3 and valid_scores[idx].item() > 0.4:
                    sim_drop = F.cosine_similarity(
                        original_prototypes[pred_cat_id].unsqueeze(0), 
                        prototypes[pred_cat_id].unsqueeze(0)
                    ).item()
                    
                    class_name = cat_id_to_name[pred_cat_id]
                    conf_score = valid_scores[idx].item()
                    
                    # 1. JSON 로그 데이터 추가
                    poisoning_logs.append({
                        "iteration": i,
                        "image_id": img_id,
                        "class_id": pred_cat_id,
                        "class_name": class_name,
                        "confidence": float(conf_score),
                        "max_iou_with_gt": float(max_iou),
                        "cosine_similarity_drop": float(sim_drop),
                        "bbox": pred_box
                    })
                    
                    # 2. 오답 시각화 이미지 저장 (빨간 박스로 표시)
                    vis_img = img_bgr_corrupted.copy()
                    x, y, w, h = map(int, pred_box)
                    cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red Box
                    label_text = f"{class_name} ({conf_score:.2f})"
                    cv2.putText(vis_img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    img_save_path = os.path.join(diag_dir, f"poison_iter{i}_img{img_id}_{class_name}.jpg")
                    cv2.imwrite(img_save_path, vis_img)

        # 100번마다 진행 상황 로그 저장
        if i % 100 == 0:
            torch.save(prototypes, os.path.join(diag_dir, f"prototypes_iter_{i}.pth"))

    # 🚨 전체 오염 로그 JSON 파일로 저장
    log_save_path = os.path.join(diag_dir, "poisoning_events_log.json")
    with open(log_save_path, 'w') as f:
        json.dump(poisoning_logs, f, indent=4)
    print(f"\nSaved {len(poisoning_logs)} poisoning events to {log_save_path}")

    # 6. mAP 평가 진행
    print("\nWriting predictions to JSON...")
    with open(res_file, 'w') as f:
        json.dump(results_list, f)
        
    print("Running COCO Evaluation...")
    if len(results_list) == 0:
        print("No objects detected across the dataset.")
        return

    cocoDt = cocoGt.loadRes(res_file)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    print("\n--- Cross-Modal PTA Evaluation Complete ---")
    print(f"mAP (IoU=0.50:0.95): {cocoEval.stats[0]:.4f}")

if __name__ == "__main__":
    main()