import sys
import torch
import io
import time

import os, zmq, json, cv2, numpy as np, base64, datetime, traceback
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from scipy import signal
from PIL import Image
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DEBUG_MODE = True
LOG_DIR = "logs"
PORT = 55552

M_cache = None


# ======================================
# 🔧 CONFIG
# ======================================
class CFG:
    folder_path = "D:/iba/POSCO_welding(2025.09.30~2025.12.30)/dinov3"
    model_path = r"D:\iba\POSCO_welding(2025.09.30~2025.12.30)\dinov3\ibakoreaSystem\segmentation\laser_segmentation\weights\20251201_epoch_009_train_0.0179_val_0.0164.pt"  # 🔥 PyTorch MLP 가중치(.pt)

    REMOVE_RATIO = 0.2
    PATCH_SIZE = 8
    IMAGE_SIZE = 768
    min_height = 60
    confidence = 0.7
    rotate_angle = 4
    plot_result = True if DEBUG_MODE else False
    bbox_center = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    WEIGHTS_PATH = "D:/iba/.../dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    MODEL_NAME = "dinov3_vits16"
    MODEL_TO_NUM_LAYERS = {
        "dinov3_vits16": 12, "dinov3_vits16plus": 12,
        "dinov3_vitb16": 12, "dinov3_vitl16": 24,
        "dinov3_vith16plus": 32, "dinov3_vit7b16": 40
    }
    n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]


# ======================================
# 🔥 LOG 함수
# ======================================
def write_log(message: str):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{today}_laser.log")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    if DEBUG_MODE:
        print(line)


# ======================================
# 🔥 PatchClassifier (Torch MLP)
# ======================================
class PatchClassifier(nn.Module):
    def __init__(self, dim=384):  # DINO vits16 → feature dim = 384
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)


# ======================================
# 🔥 DINO + Torch MLP 로드
# ======================================
def model_load():
    # ---- 1) MLP classifier 로드 ----
    clf = PatchClassifier(dim=384).to(CFG.device)
    state_dict = torch.load(CFG.model_path, map_location=CFG.device)
    clf.load_state_dict(state_dict)
    clf.eval()

    # ---- 2) DINO 모델 로드 ----
    model = torch.hub.load(
        repo_or_dir=CFG.folder_path,
        model=CFG.MODEL_NAME,
        source="local",
        weights=CFG.WEIGHTS_PATH
    )

    use_fp16 = torch.cuda.is_available()
    model = model.eval().to(CFG.device)

    if use_fp16:
        model = model.half()
        write_log("🚀 FP16 모드 ON")
    else:
        write_log("💻 CPU FP32 실행")

    # warm-up
    try:
        dummy = torch.randn(1, 3, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE, device=CFG.device)
        if use_fp16:
            dummy = dummy.half()
        with torch.inference_mode():
            _ = model(dummy)
        write_log("🔥 모델 warm-up 완료")
    except:
        write_log("warm-up 에러 발생")

    return model, clf, CFG.device, use_fp16


# ======================================
# 🔥 BASE64 decoder
# ======================================
def decode_base64_image(img_b64):
    try:
        img_bytes = base64.b64decode(img_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        write_log(f"❌ 이미지 디코딩 실패: {e}")
        return None


# ======================================
# 🔥 IMAGE INFERENCE (MAIN)
# ======================================
def main():
    model, clf, device, use_fp16 = model_load()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")

    write_log(f"✅ ZMQ 서버 실행됨 (포트 {PORT})")

    while True:
        try:
            frames = socket.recv_multipart()
            file_name = frames[0].decode()
            img_bytes = frames[1]

            write_log(f"📩 요청 수신: {file_name}")

            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image_np is None:
                raise ValueError("이미지 디코딩 실패")

            result = infer_image_one(
                model, clf, device, use_fp16,
                image_np,
                rotate_angle=CFG.rotate_angle,
                plot_result=CFG.plot_result
            )

            if isinstance(result, tuple):
                upper_x, lower_x = result
                socket.send_string(json.dumps({
                    "status": "검출 완료",
                    "upper_x": round(upper_x),
                    "lower_x": round(lower_x)
                }))
            else:
                socket.send_string(json.dumps({
                    "status": "검출 실패",
                    "upper_x": -1,
                    "lower_x": -1
                }))

        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)}"
            write_log(msg)
            traceback.print_exc()
            socket.send_string(json.dumps({"status": "ERROR", "msg": msg}))


# ======================================
# 🔥 IMAGE → Δx 계산
# ======================================
def infer_image_one(model, clf, device, use_fp16, test_image,
                    rotate_angle=1.3, plot_result=False):
    start_time = time.time()
    img_np = np.array(test_image)
    h, w = img_np.shape[:2]

    # --- 회전 ---
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
    rotated = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    upper_img = rotated[:h // 2, :]
    lower_img = rotated[h // 2:, :]

    upper_x, upper_vis = detect_single_laser(model, clf, device, use_fp16, upper_img, CFG.min_height)
    lower_x, lower_vis = detect_single_laser(model, clf, device, use_fp16, lower_img, CFG.min_height)

    if upper_x < 0 or lower_x < 0:
        write_log("❌ 레이저 검출 실패")
        return -1, -1

    dx = lower_x - upper_x
    write_log(f"✔ upper={upper_x}, lower={lower_x}, Δx={dx} px")
    write_log(f"⏱ time={time.time() - start_time:.3f}s")

    if plot_result:
        visualize_upper_lower(rotated, upper_vis, lower_vis, upper_x, lower_x, f"{dx}px", rotate_angle)

    return upper_x, lower_x


# ======================================
# 🔥 레이저 1개 탐지
# ======================================
def detect_single_laser(model, clf, device, use_fp16, img_np, min_height=80):
    h, w = img_np.shape[:2]
    pil_img = Image.fromarray(img_np)

    # --- DINO 전처리 ---
    img_resized = resize_transform(pil_img)
    img_tensor = TF.normalize(img_resized, mean=CFG.IMAGENET_MEAN, std=CFG.IMAGENET_STD)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    if use_fp16:
        img_tensor = img_tensor.half()

    # --- DINO feature ---
    with torch.inference_mode():
        feats = model.get_intermediate_layers(img_tensor,
                                              n=range(CFG.n_layers),
                                              reshape=True,
                                              norm=True)
        x = feats[-1].squeeze().detach().cpu()
        dim, Hf, Wf = x.shape
        x = x.view(dim, -1).permute(1, 0)

    # --- Torch classifier (sigmoid) ---
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    # if use_fp16: # classifier는 fp36 유지
    #     x_tensor = x_tensor.half()
    x_tensor = x_tensor.float()

    with torch.no_grad():
        logits = clf(x_tensor)
        fg_score = torch.sigmoid(logits).squeeze(1).cpu().numpy()

    fg_score = fg_score.reshape(Hf, Wf)
    fg_score_mf = signal.medfilt2d(fg_score, kernel_size=3)

    # --- upsample to original size ---
    fg_up = cv2.resize(fg_score_mf, (w, h), interpolation=cv2.INTER_CUBIC)

    mask = (fg_up > CFG.confidence).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean)

    # --- 가장 큰 blob 선택 ---
    best_label = -1
    best_area = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            best_label = label

    if best_label == -1:
        return -1, img_np

    x0, y0, w0, h0 = stats[best_label, cv2.CC_STAT_LEFT], stats[best_label, cv2.CC_STAT_TOP], \
        stats[best_label, cv2.CC_STAT_WIDTH], stats[best_label, cv2.CC_STAT_HEIGHT]

    if h0 < min_height:
        write_log(f"⚠ 너무 작은 blob: h={h0}")
        return -1, img_np

    vis = img_np.copy()

    # --- 중심 ---
    if CFG.bbox_center:
        cx = x0 + w0 / 2
        cy = y0 + h0 / 2
    else:
        ys, xs = np.where(labels == best_label)
        weights = fg_up[labels == best_label]
        cx = (xs * weights).sum() / weights.sum()
        cy = (ys * weights).sum() / weights.sum()

    cx = int(cx)

    cv2.rectangle(vis, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 255), 2)
    cv2.line(vis, (cx, y0), (cx, y0 + h0), (0, 0, 255), 2)
    cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)

    return cx, vis


# ======================================
# 🔥 RESIZE helper
# ======================================
def resize_transform(img: Image, image_size=CFG.IMAGE_SIZE, patch_size=CFG.PATCH_SIZE):
    w, h = img.size
    Hp = int(image_size / patch_size)
    Wp = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(img, (Hp * patch_size, Wp * patch_size)))


# ======================================
# 🔥 시각화
# ======================================
def visualize_upper_lower(rotated, upper_vis, lower_vis, upper_x, lower_x, dx_text, rotate_angle):
    h, w = rotated.shape[:2]

    merged = rotated.copy()
    merged[:h // 2] = upper_vis
    merged[h // 2:] = lower_vis

    merged = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)
    rot_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(rot_rgb)
    plt.title(f"Rotate {rotate_angle}°")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(merged)
    plt.title(f"Laser Detect Δx={dx_text}")
    plt.axis("off")

    plt.show()


# ======================================
# RUN
# ======================================
if __name__ == "__main__":
    main()
