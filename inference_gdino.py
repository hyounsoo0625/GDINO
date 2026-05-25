import os
import sys
import cv2
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

# Hugging Face Transformers
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 상대 경로를 통한 모듈 임포트 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 유저가 제공한 데이터 준비 모듈 임포트 (파일명이 data_prepare.py 인 경우)
# 만약 data_prepare 폴더 안의 coco_c.py 라면 from data_prepare.coco_c import apply_corruption 로 수정하세요.
try:
    from data_prepare import apply_corruption
except ImportError:
    print("Warning: Cannot import apply_corruption from data_prepare. Ensure the path is correct.")
    # Fallback dummy function if import fails during setup
    def apply_corruption(img_bgr, corruption_name, severity=3):
        return img_bgr

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Grounding DINO on COCO-C")
    parser.add_argument('--corruption', type=str, default='gaussian_noise', 
                        help='Name of the corruption to apply (e.g., gaussian_noise, motion_blur). Use "clean" for original images.')
    parser.add_argument('--severity', type=int, default=3, choices=[1,2,3,4,5], 
                        help='Severity level of the corruption (1-5)')
    parser.add_argument('--box_threshold', type=float, default=0.4, help='Bounding box confidence threshold')
    parser.add_argument('--text_threshold', type=float, default=0.3, help='Text match confidence threshold')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save prediction JSON')
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. 경로 설정
    ann_file = os.path.join('data', 'coco', 'annotations', 'instances_val2017.json')
    img_dir = os.path.join('data', 'coco', 'val2017')
    os.makedirs(args.output_dir, exist_ok=True)
    
    res_file = os.path.join(args.output_dir, f"gdino_preds_{args.corruption}_s{args.severity}.json")
    
    # 2. COCO 데이터셋 로드
    print("Loading COCO annotations...")
    cocoGt = COCO(ann_file)
    img_ids = cocoGt.getImgIds()
    
    # 카테고리 정보 추출 및 매핑 딕셔너리 생성
    categories = cocoGt.loadCats(cocoGt.getCatIds())
    cat_name_to_id = {cat['name'].lower(): cat['id'] for cat in categories}
    
    # Grounding DINO 프롬프트 생성 (예: "person . bicycle . car . ...")
    text_prompt = " . ".join([cat['name'].lower() for cat in categories]) + " ."
    print(f"Text Prompt for Grounding DINO:\n{text_prompt[:100]}...\n")

    # 3. 모델 로드
    print("Loading Grounding DINO model...")
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, device_map="auto")
    model.eval()

    results_list = []
    
    # 4. 추론 루프
    print(f"Evaluating on {len(img_ids)} images with {args.corruption} (Severity: {args.severity})...")
    for img_id in tqdm(img_ids):
        img_info = cocoGt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"Missing image: {img_path}")
            continue
            
        # 이미지 읽기 및 Corruption 적용
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
            
        if args.corruption.lower() != 'clean':
            img_bgr = apply_corruption(img_bgr, corruption_name=args.corruption, severity=args.severity)
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # 모델 입력 전처리
        inputs = processor(images=pil_img, text=text_prompt, return_tensors="pt").to(model.device)
        
        # 추론
        with torch.no_grad():
            outputs = model(**inputs)
            
        # 후처리 (NMS 및 Threshold 적용)
        target_sizes = [pil_img.size[::-1]]
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=target_sizes
        )[0]
        
        # 5. 결과를 COCO 형식으로 변환
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            # DINO는 라벨을 리스트 형태의 텍스트로 반환함
            clean_label = label.strip().lower()
            
            # 예측된 텍스트가 COCO 카테고리에 있는지 확인
            if clean_label in cat_name_to_id:
                cat_id = cat_name_to_id[clean_label]
                
                # BBox 변환: [x1, y1, x2, y2] -> [x, y, width, height]
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1
                
                results_list.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "score": round(score.item(), 4)
                })

    # 6. 예측 결과 저장
    print(f"\nSaving predictions to {res_file}...")
    with open(res_file, 'w') as f:
        json.dump(results_list, f)

    # 7. mAP 평가 (COCOeval)
    if len(results_list) == 0:
        print("No objects detected. mAP is 0.0")
        return

    print("\nRunning COCO Evaluation...")
    cocoDt = cocoGt.loadRes(res_file)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    print("\n" + "="*50)
    print(f"Evaluation Summary for {args.corruption} (Severity {args.severity})")
    print("="*50)
    print(f"mAP (IoU=0.50:0.95) : {cocoEval.stats[0]:.4f}")
    print(f"mAP (IoU=0.50)      : {cocoEval.stats[1]:.4f}")
    print(f"mAP (IoU=0.75)      : {cocoEval.stats[2]:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()