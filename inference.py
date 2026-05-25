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

def initialize_text_prototypes(model, processor, categories):
    """
    각 카테고리의 텍스트를 Grounding DINO의 Text Backbone과 Projection Head에 통과시켜
    256차원의 텍스트 피쳐를 추출하고, 이를 초기 Prototype으로 설정합니다.
    """
    print("Initializing Text-driven Prototypes...")
    prototypes = {}
    
    for cat in categories:
        cat_name = cat['name']
        cat_id = cat['id']
        
        # 텍스트 전처리 (단일 클래스 이름)
        inputs = processor(text=[cat_name], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            # 1. BERT Text Backbone 통과 (768 차원)
            text_outputs = model.model.text_backbone(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            # [CLS] 토큰의 임베딩 추출 (batch_size, hidden_size)
            cls_features = text_outputs.last_hidden_state[:, 0, :]
            
            # 2. Text Projection 통과 (256 차원 Joint Space로 변환)
            # 바로 이 층이 시각적 Object Query와 만나기 직전의 텍스트 헤드입니다.
            projected_features = model.model.text_projection(cls_features)
            
            # L2 정규화
            projected_features = F.normalize(projected_features, p=2, dim=-1)
            
        prototypes[cat_id] = projected_features.squeeze(0).clone()
        
    return prototypes

def extract_object_queries(outputs, valid_boxes, target_size):
    """
    추론된 Bounding Box를 기반으로 해당 객체를 예측한 원본 Object Query를 추출합니다.
    """
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
        
        # Visual Query 추출 및 정규화
        query = object_queries[best_idx]
        query = F.normalize(query, p=2, dim=-1)
        extracted_queries.append(query)
        
    if len(extracted_queries) == 0:
        return torch.empty(0, object_queries.size(-1)).to(object_queries.device)
        
    return torch.stack(extracted_queries)
def calculate_iou(boxA, boxB):
    # box format: [x, y, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)
    
def update_instance_prototypes(query_features, probs, labels, prototypes, alpha=0.01, T=20):
    """
    Object Query(비주얼)로 Text Prototype(텍스트)를 업데이트하는 Cross-modal PTA.
    """
    for i, label_idx in enumerate(labels):
        w = probs[i]
        
        if w >= 1e-1:
            w_new = 1 - torch.exp(-w / T)
            
            current_proto = prototypes[label_idx]
            feat = query_features[i]
            
            # 비주얼 특징으로 텍스트 특징을 보정 (EMA)
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
    
    # 🌟 핵심 변경 포인트: 모델의 Text Head를 통과시킨 임베딩으로 Prototype 초기화
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
            outputs,
            inputs.input_ids,
            threshold=0.4, 
            text_threshold=0.3,
            target_sizes=target_sizes
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
        
        # --- PTA CROSS-MODAL PROTOTYPE UPDATE ---
        if len(valid_indices) > 0:
            valid_boxes = boxes[valid_indices]
            valid_scores = scores[valid_indices]
            
            # 디코더를 통과한 시각적 특징(Visual Query) 추출
            query_features = extract_object_queries(
                outputs=outputs, 
                valid_boxes=valid_boxes, 
                target_size=pil_img.size
            )
            
            # 비주얼 특징으로 텍스트 프로토타입 업데이트
            prototypes = update_instance_prototypes(
                query_features=query_features, 
                probs=valid_scores, 
                labels=detected_cat_ids, 
                prototypes=prototypes,
                alpha=0.01, 
                T=20
            )

        if i % 100 == 0:
            torch.save(prototypes, f"./prototypes_iter_{i}.pth")

    print("Writing predictions to JSON...")
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