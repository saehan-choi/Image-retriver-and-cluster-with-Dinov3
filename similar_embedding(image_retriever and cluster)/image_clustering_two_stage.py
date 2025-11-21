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

    target_folder = r"C:\4-2CAL_welder\dataset\Backup_현장\okng_datasets\NG\NG_images"
    batch_size = 64

    cluster_output = r"C:\4-2CAL_welder\dataset\Backup_현장\okng_datasets\NG\NG_cluseters"

    # 1차 & 2차 파라미터 분리
    min_cluster_size_1 = 10
    min_samples_1 = 3

    min_cluster_size_2 = 5
    min_samples_2 = 2

    # 몇 장 이상이면 2차 클러스터링 수행할지 (임계값)
    second_pass_threshold = 50


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
                x = out.get("x_norm_gettoken", list(out.values())[0])
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

        if len(batch_tensors) == batch_size:
            batch = torch.stack(batch_tensors).to(CFG.device)
            with torch.no_grad():
                z = model(batch)
                z = F.normalize(z, dim=1)
            embeddings.append(z.cpu())
            batch_tensors = []

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
# HDBSCAN 클러스터링 함수
# =====================================================================
def run_hdbscan(embs_np, min_cluster_size, min_samples):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean"
    )
    return clusterer.fit_predict(embs_np)


# =====================================================================
# 2단계 클러스터링 실행
# =====================================================================
def second_pass_cluster(cluster_path):
    print(f"\n🔥 2차 클러스터링 수행: {cluster_path}")

    img_paths = load_images(cluster_path)
    if len(img_paths) < CFG.min_cluster_size_2:
        print("2차 클러스터링 스킵 (이미지 너무 적음)")
        return

    # 모델 로드
    model = DinoEmbedder().to(CFG.device)
    model.eval()

    # 임베딩 계산
    embs = get_embeddings_batch(model, img_paths, batch_size=CFG.batch_size)
    labels = run_hdbscan(
        embs.numpy(),
        CFG.min_cluster_size_2,
        CFG.min_samples_2
    )

    # 결과 저장
    for idx, label in enumerate(labels):
        subfolder = os.path.join(cluster_path, f"subcluster_{label}")
        os.makedirs(subfolder, exist_ok=True)

        src = img_paths[idx]
        dst = os.path.join(subfolder, os.path.basename(src))
        shutil.copy(src, dst)

    print("2차 완료!")


# =====================================================================
# 1단계 + 2단계 전체 파이프라인
# =====================================================================
def cluster_ng_images():
    img_paths = load_images(CFG.target_folder)
    print(f"총 이미지 수: {len(img_paths)}")

    if len(img_paths) == 0:
        print("⚠ 이미지 없음")
        return

    print("\n⭐ 1차 모델 로딩 중...")
    model = DinoEmbedder().to(CFG.device)
    model.eval()

    print("\n⭐ 1차 임베딩 추출 중...")
    embs = get_embeddings_batch(model, img_paths, batch_size=CFG.batch_size)
    labels = run_hdbscan(
        embs.numpy(),
        CFG.min_cluster_size_1,
        CFG.min_samples_1
    )

    # 1차 결과 저장
    os.makedirs(CFG.cluster_output, exist_ok=True)

    cluster_members = {}  # {cluster_id: [img_paths]}

    for idx, label in enumerate(labels):
        cluster_members.setdefault(label, []).append(img_paths[idx])

    print("\n📦 1차 클러스터 저장")

    # 1차 결과 저장
    for label, paths in cluster_members.items():
        cluster_dir = os.path.join(CFG.cluster_output, f"cluster_{label}")
        os.makedirs(cluster_dir, exist_ok=True)
        for p in paths:
            shutil.copy(p, os.path.join(cluster_dir, os.path.basename(p)))

        # 2차 클러스터링 대상인지 확인
        if len(paths) >= CFG.second_pass_threshold and label != -1:
            second_pass_cluster(cluster_dir)

    print("\n🎉 전체 파이프라인 완료!")


# =====================================================================
# 실행
# =====================================================================
if __name__ == "__main__":
    cluster_ng_images()
