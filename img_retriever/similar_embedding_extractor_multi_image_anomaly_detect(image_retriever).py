import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import albumentations as A
import shutil
import hashlib
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

    ok_folder = r"C:\4-2CAL_welder\dataset\Backup_현장\okng_datasets\OK"
    sample_folder = r"C:\4-2CAL_welder\dataset\Backup_현장\okng_datasets\OK\sample"
    ng_output_folder = r"C:\4-2CAL_welder\dataset\Backup_현장\okng_datasets\NG\NG_images"

    batch_size = 64
    top_k = 10000

    OK_CACHE_FILE = r"C:\4-2CAL_welder\PythonPlugin\AI_Model\weights\ok_embeddings_cache_anomaly_maha.pt"
    SAMPLE_CACHE_FILE = r"C:\4-2CAL_welder\PythonPlugin\AI_Model\weights\sample_embeddings_cache_anomaly_maha.pt"

    USE_FIRST = 10000000  # 장수 조절할때 사용 (이미지 너무 많아서 오래걸릴때)

    move_results_to_NG_folder = True

    # centroid 적용된 것임
    search_NG = True # 이거 True로 하면 NG만 찾고, False로 하면 OK만 찾음 (True시 가장 anomaly detect 하고, False시 anomaly detect 한 score의 가장 낮은것들을 찾아냄)



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
# 이미지 로딩
# =====================================================================
def imread(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


transform = A.Compose([
    A.Resize(*CFG.img_resize),
    A.Normalize(),
    ToTensorV2(),
])



# =====================================================================
# md5 hash
# =====================================================================
def md5_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()



# =====================================================================
# Batch Embedding
# =====================================================================
def get_embeddings_batch_safe(model, paths, batch_size=32):
    embeddings = []
    batch_tensors = []

    for p in tqdm(paths, desc="Embedding batch"):
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
# 폴더 이미지 로딩
# =====================================================================
def load_images(folder):
    exts = (".bmp", ".png", ".jpg", ".jpeg")
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ])



# =====================================================================
# Smart Cache 공통 처리 (OK + SAMPLE 용)
# =====================================================================
def load_or_update_cache(model, image_files, cache_file):
    if not os.path.exists(cache_file):
        print(f"\n⚡ 캐시 없음 → {cache_file} 생성 중…")
        embs = get_embeddings_batch_safe(model, image_files, CFG.batch_size)
        hashes = {p: md5_hash(p) for p in image_files}
        torch.save({"paths": image_files, "embeddings": embs.cpu(), "hashes": hashes}, cache_file)
        return image_files, embs.to(CFG.device)

    # 기존 캐시 로드
    print(f"\n📦 캐시 로딩({cache_file})…")
    cache = torch.load(cache_file)
    cached_paths = cache["paths"]
    cached_embs = cache["embeddings"]
    cached_hashes = cache["hashes"]

    cached_set = set(cached_paths)
    current_set = set(image_files)

    # 삭제된 파일 제거
    deleted = cached_set - current_set
    if deleted:
        keep_idx = [i for i, p in enumerate(cached_paths) if p not in deleted]
        cached_paths = [cached_paths[i] for i in keep_idx]
        cached_embs = cached_embs[keep_idx]
        cached_hashes = {p: cached_hashes[p] for p in cached_paths}

    # 신규 추가 파일
    added = current_set - cached_set
    if added:
        print(f"➕ {len(added)}개 추가됨 → 임베딩 계산")
        added = list(added)
        new_embs = get_embeddings_batch_safe(model, added, CFG.batch_size)
        cached_paths.extend(added)
        cached_embs = torch.cat([cached_embs, new_embs.cpu()], dim=0)
        for p in added:
            cached_hashes[p] = md5_hash(p)

    # 변경된 파일 재계산
    changed = []
    for p in cached_paths:
        if cached_hashes[p] != md5_hash(p):
            changed.append(p)

    if changed:
        print(f"🔄 {len(changed)}개 변경됨 → 재계산")
        new_embs = get_embeddings_batch_safe(model, changed, CFG.batch_size)
        for i, p in enumerate(cached_paths):
            if p in changed:
                idx = changed.index(p)
                cached_embs[i] = new_embs[idx]
                cached_hashes[p] = md5_hash(p)

    torch.save({"paths": cached_paths, "embeddings": cached_embs, "hashes": cached_hashes}, cache_file)

    return cached_paths, cached_embs.to(CFG.device)



# =====================================================================
# Mahalanobis Distance
# =====================================================================
def mahalanobis_distance(x, mean, inv_cov):
    diff = x - mean
    return torch.sqrt(torch.dot(diff, inv_cov @ diff)).item()



# =====================================================================
# 🚀 Mahalanobis 기반 anomaly filtering
# =====================================================================
def run_anomaly_filtering():
    print("==============================")
    print("🔥 Mahalanobis ANOMALY DETECT")
    print("==============================")

    # Load OK images
    ok_paths = load_images(CFG.ok_folder)[:CFG.USE_FIRST]
    print(f"OK image count: {len(ok_paths)}")

    # Load Sample
    sample_paths = load_images(CFG.sample_folder)
    print(f"SAMPLE count: {len(sample_paths)}")

    # model
    model = DinoEmbedder().to(CFG.device)
    model.eval()

    # OK Cache
    ok_paths, ok_embs = load_or_update_cache(model, ok_paths, CFG.OK_CACHE_FILE)

    # centroid + cov inverse
    centroid = torch.mean(ok_embs, dim=0)

    # 공분산 계산
    cov = torch.cov(ok_embs.T)

    # 안정성을 위한 small identity (GPU로!)
    eps_I = 1e-6 * torch.eye(ok_embs.shape[1], device=CFG.device)

    cov = cov + eps_I

    # invert
    inv_cov = torch.inverse(cov)

    # 디바이스 보정
    centroid = centroid.to(CFG.device)
    inv_cov = inv_cov.to(CFG.device)

    centroid = centroid.to(CFG.device)
    inv_cov = inv_cov.to(CFG.device)

    # SAMPLE Cache
    sample_paths, sample_embs = load_or_update_cache(model, sample_paths, CFG.SAMPLE_CACHE_FILE)

    # anomaly score = Mahalanobis
    results = []
    for i, emb in enumerate(sample_embs):
        score = mahalanobis_distance(emb.to(CFG.device), centroid, inv_cov)
        results.append((sample_paths[i], score))

    results_sorted = sorted(results, key=lambda x: x[1], reverse=CFG.search_NG)

    print("\n===== 🔥 NG 의심 Top-K (Mahalanobis) =====")
    for p, s in results_sorted[:CFG.top_k]:
        print(f"{s:.3f}   {p}")

    # move files
    if CFG.move_results_to_NG_folder:
        for p, s in results_sorted[:CFG.top_k]:
            try:
                dst = os.path.join(CFG.ng_output_folder, os.path.basename(p))
                shutil.move(p, dst)
                print("➡ 이동:", p)
            except Exception as e:
                print("⚠ 이동 실패:", e)

    return results_sorted



# =====================================================================
# ▶ 실행
# =====================================================================
if __name__ == "__main__":
    run_anomaly_filtering()