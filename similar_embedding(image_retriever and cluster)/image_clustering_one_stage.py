import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import albumentations as A
import shutil
import hdbscan

from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


# =====================================================================
# ⚙️ 설정
# =====================================================================
class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dinov3_location = "C:/dinov3/"
    model_name = "dinov3_vits16"
    dinov3_weights_path = "C:/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

    img_resize = (800, 800)

    # ⭐ NG 이미지 폴더 전체 클러스터링
    target_folder = r"C:\4-2CAL_welder\dataset\Backup_현장\okng_datasets\NG\NG_images"

    batch_size = 64

    # 출력 위치
    cluster_output = r"C:\4-2CAL_welder\dataset\Backup_현장\okng_datasets\NG\NG_cluseters"


# =====================================================================
# 🧠 DINOv3 임베딩 모델
# =====================================================================
class DinoEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            repo_or_dir=CFG.dinov3_location,
            model=CFG.model_name,
            source="local",
            weights=CFG.dinov3_weights_path,
        )

    def forward(self, x):
        with torch.no_grad():
            out = self.model.forward_features(x)
            if isinstance(out, dict):
                x = out.get("x_norm_clstoken", list(out.values())[0])
            else:
                x = out
        return x


# =====================================================================
# 이미지 로딩 함수
# =====================================================================
def imread(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# Albumentations transform
transform = A.Compose([
    A.Resize(*CFG.img_resize),
    A.Normalize(),
    ToTensorV2(),
])


# =====================================================================
# 배치 단위 임베딩
# =====================================================================
def get_embeddings_batch(model, paths, batch_size=32):
    embeddings = []
    batch_tensors = []

    for p in tqdm(paths, desc="Embedding"):
        img = imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(image=img)["image"]
        batch_tensors.append(tensor)

        # 배치 처리
        if len(batch_tensors) == batch_size:
            batch = torch.stack(batch_tensors).to(CFG.device)
            with torch.no_grad():
                z = model(batch)
                z = F.normalize(z, dim=1)
            embeddings.append(z.cpu())
            batch_tensors = []

    # 마지막 배치 처리
    if len(batch_tensors) > 0:
        batch = torch.stack(batch_tensors).to(CFG.device)
        with torch.no_grad():
            z = model(batch)
            z = F.normalize(z, dim=1)
        embeddings.append(z.cpu())

    return torch.cat(embeddings, dim=0)


# =====================================================================
# 폴더에서 이미지 불러오기
# =====================================================================
def load_images(folder):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ])


# =====================================================================
# 클러스터링 실행
# =====================================================================
def cluster_ng_images():
    # 이미지 경로들
    img_paths = load_images(CFG.target_folder)
    print(f"총 이미지 수: {len(img_paths)}")

    if len(img_paths) == 0:
        print("⚠ 이미지 없음")
        return

    # 모델 로드
    print("\n⭐ DINOv3 로딩 중...")
    model = DinoEmbedder().to(CFG.device)
    model.eval()

    # 임베딩 추출
    print("\n⭐ 임베딩 추출 중...")
    embs = get_embeddings_batch(model, img_paths, batch_size=CFG.batch_size)
    embs_np = embs.numpy()

    # 클러스터링
    print("\n⭐ HDBSCAN 클러스터링 중...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=3
    )
    labels = clusterer.fit_predict(embs_np)

    # 출력 폴더 생성
    os.makedirs(CFG.cluster_output, exist_ok=True)

    print("\n📦 클러스터 결과 정리 중...")

    for idx, label in enumerate(labels):
        cluster_dir = os.path.join(CFG.cluster_output, f"cluster_{label}")
        os.makedirs(cluster_dir, exist_ok=True)

        src = img_paths[idx]
        dst = os.path.join(cluster_dir, os.path.basename(src))
        shutil.copy(src, dst)

    print("\n🎉 완료! 클러스터링된 폴더가 생성되었습니다.")


# =====================================================================
# 실행
# =====================================================================
if __name__ == "__main__":
    cluster_ng_images()
