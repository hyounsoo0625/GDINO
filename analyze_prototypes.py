import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from scipy.ndimage import laplace
from pycocotools.coco import COCO

def load_all_checkpoints(base_dir="./"):
    """저장된 모든 pth 파일을 정렬하여 로드합니다."""
    files = glob.glob(os.path.join(base_dir, "prototypes_iter_*.pth"))
    if not files:
        raise FileNotFoundError("프로토타입 .pth 파일을 찾을 수 없습니다.")
    files.sort(key=lambda x: int(x.split('_iter_')[-1].split('.pth')[0]))
    
    iterations = [int(f.split('_iter_')[-1].split('.pth')[0]) for f in files]
    checkpoints = [torch.load(f, map_location='cpu') for f in files]
    return iterations, checkpoints

def get_coco_labels():
    """COCO 카테고리 이름을 가져옵니다."""
    ann_file = './data/coco/annotations/instances_val2017.json'
    if os.path.exists(ann_file):
        coco = COCO(ann_file)
        categories = coco.loadCats(coco.getCatIds())
        return {cat['id']: cat['name'] for cat in categories}
    return None

def main():
    print("데이터 및 체크포인트 로드 중...")
    iterations, checkpoints = load_all_checkpoints()
    cat_id_to_name = get_coco_labels()
    
    # 한번이라도 업데이트가 일어난 유효 클래스 ID 찾기
    last_ckpt = checkpoints[-1]
    active_cat_ids = [cid for cid, proto in last_ckpt.items() if proto.norm() > 0]
    
    if not active_cat_ids:
        print("업데이트된 프로토타입이 없습니다.")
        return

    # -------------------------------------------------------------------------
    # 1. Prototype별 t-SNE 시각화 (마지막 Iteration 기준 클래스 간 분리도)
    # -------------------------------------------------------------------------
    print("[1/3] t-SNE 시각화 진행 중...")
    features = []
    labels = []
    
    for cid in active_cat_ids:
        features.append(last_ckpt[cid].numpy())
        class_name = cat_id_to_name[cid] if cat_id_to_name else f"Class_{cid}"
        labels.append(class_name)
        
    features = np.array(features)
    
    # 80개 클래스 전체를 그리면 복잡하므로 상위 몇 개 혹은 perplexity 조절
    perplexity = min(30, len(features) - 1)
    if perplexity >= 1:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        tsne_results = tsne.fit_transform(features)
        
        plt.figure(figsize=(10, 8), dpi=300)
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], cmap='tab20', alpha=0.7)
        
        # 대표적인 몇 개 클래스만 텍스트 라벨링 (가독성 확보)
        for idx, label in enumerate(labels):
            if idx % 1 == 0:  # 3개 중 1개만 라벨 표시 (겹침 방지)
                plt.annotate(label, (tsne_results[idx, 0], tsne_results[idx, 1]), fontsize=9, fontweight='bold')
                
        plt.title("t-SNE Visualization of Final Class Prototypes", fontsize=14, fontweight='bold')
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("analysis_1_tsne.png", bbox_inches='tight')
        print("-> 'analysis_1_tsne.png' 저장 완료")

    # -------------------------------------------------------------------------
    # 2. 각 Class별 Prototype의 특징 시각화 (Feature Map Re-construction)
    # -------------------------------------------------------------------------
    print("[2/3] 프로토타입 시각적 특징 복원 중...")
    # 논문에 넣을 핵심 타겟 클래스 지정 (예: person, car, dog 등 데이터가 많은 클래스)
    target_classes = active_cat_ids[:10] # 앞에서부터 4개 선택
    
    fig, axes = plt.subplots(1, len(target_classes), figsize=(4 * len(target_classes), 4), dpi=300)
    if len(target_classes) == 1: axes = [axes]
        
    for idx, cid in enumerate(target_classes):
        proto_vector = last_ckpt[cid]
        
        # 1D 벡터를 다시 이미지 공간 (3, 224, 224) -> (224, 224, 3)으로 변환
        proto_img = proto_vector.view(3, 224, 224).permute(1, 2, 0).numpy()
        
        # Min-Max Normalization (시각화를 위해 0~1 사이로 변경)
        proto_img = (proto_img - proto_img.min()) / (proto_img.max() - proto_img.min() + 1e-8)
        
        axes[idx].imshow(proto_img)
        class_name = cat_id_to_name[cid] if cat_id_to_name else f"Class_{cid}"
        axes[idx].set_title(f"Proto: {class_name}", fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        
    plt.suptitle("Visualized Class Prototypes (Accumulated Features)", fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig("analysis_2_prototype_features.png", bbox_inches='tight')
    print("-> 'analysis_2_prototype_features.png' 저장 완료")

    # -------------------------------------------------------------------------
    # 3. Prototype의 Noise 정도 시각화 (Total Variation / Laplacian Variance)
    # -------------------------------------------------------------------------
    print("[3/3] 시간에 따른 노이즈(고주파 성분) 변화량 분석 중...")
    
    plt.figure(figsize=(9, 5), dpi=300)
    
    # 분석할 대표 클래스 3개 선택
    noise_targets = active_cat_ids[:3]
    
    for cid in noise_targets:
        noise_scores = []
        
        for ckpt in checkpoints:
            proto_vector = ckpt[cid]
            if proto_vector.norm() == 0:
                noise_scores.append(0.0)
                continue
                
            # 이미지 형태로 복원
            img = proto_vector.view(3, 224, 224).permute(1, 2, 0).numpy()
            # 그레이스케일 변환 (정밀한 노이즈 계산용)
            gray_img = np.mean(img, axis=2)
            
            # Laplacian 필터를 적용하여 고주파 성분(노이즈 및 경계선) 추출
            lap_var = np.var(laplace(gray_img))
            noise_scores.append(lap_var)
            
        class_name = cat_id_to_name[cid] if cat_id_to_name else f"Class_{cid}"
        plt.plot(iterations, noise_scores, marker='x', linewidth=2, label=f"{class_name} Noise level")
        
    plt.title("Prototype Noise Accumulation Over Iterations", fontsize=13, fontweight='bold')
    plt.xlabel("Iteration")
    plt.ylabel("Noise Variance (Laplacian Variance)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("analysis_3_noise_trend.png", bbox_inches='tight')
    print("-> 'analysis_3_noise_trend.png' 저장 완료")

if __name__ == "__main__":
    main()