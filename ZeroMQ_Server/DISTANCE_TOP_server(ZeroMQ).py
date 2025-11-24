import sys
import pickle
import torch
import io
import time

import os, zmq, json, cv2, numpy as np, base64, datetime, traceback
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from scipy import signal
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


DEBUG_MODE = True
LOG_DIR = "logs"
PORT = 55552

M_cache = None

# ✅ 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
# ✅ 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False



# ==========================
# 🔧 설정
# ==========================
class CFG:
    folder_path = "C:/dinov3"
    model_path = "C:/dinov3/ibakoreaSystem/segmentation/laser_segmentation/weights/polygon_classifier.pkl"
    test_dir = rf"C:/Users/레노버/Downloads/Backup_현장/Backup_20250716/TOP_20250716142904" # 단차 차이 좀 나는거

    # 전처리 (중간 부분 제거)
    REMOVE_RATIO = 0.2

    PATCH_SIZE = 8
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    WEIGHTS_PATH = "C:/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

    MODEL_TO_NUM_LAYERS = {
        "dinov3_vits16": 12,
        "dinov3_vits16plus": 12,
        "dinov3_vitb16": 12,
        "dinov3_vitl16": 24,
        "dinov3_vith16plus": 32,
        "dinov3_vit7b16": 40,
    }
    MODEL_NAME = "dinov3_vits16"
    n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]
    IMAGE_SIZE = 768
    plot_result = True if DEBUG_MODE else False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

def write_log(message: str):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{today}.log")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    if DEBUG_MODE:
        print(line)


def decode_base64_image(img_b64):
    try:
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img
    except Exception as e:
        write_log(f"❌ 이미지 디코딩 실패: {e}")
        return None


def model_load():
    with open(CFG.model_path, "rb") as f:
        clf = pickle.load(f)

    model = torch.hub.load(repo_or_dir=CFG.folder_path, model=CFG.MODEL_NAME, source="local", weights=CFG.WEIGHTS_PATH)
    use_fp16 = torch.cuda.is_available()
    model = model.eval().to(CFG.device)
    if use_fp16:
        model = model.half()
        print("🚀 GPU + FP16 모드로 실행 중")
    else:
        print("💻 CPU 모드로 실행 중 (FP32)")

    try:
        dummy = torch.randn(1, 3, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE, device=CFG.device)
        if use_fp16:
            dummy = dummy.half()
        with torch.inference_mode():
            _ = model(dummy)
        print("🔥 모델 warm-up 완료")
    except Exception as e:
        print(f"⚠️ warm-up 중 오류 발생: {e}")

    return model, clf, CFG.device, use_fp16



def main():
    model, clf, device, use_fp16 = model_load()
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")

    print(f"✅ ZeroMQ 서버 대기 중... (tcp://*:{PORT})")

    while True:
        try:
            msg = socket.recv_string()
            req = json.loads(msg)

            cmd = req.get("cmd", "")
            rotate_angle = float(req.get("rotate_angle", 1.3))
            remove_ratio = float(req.get("remove_ratio", 0.15))

            # 만약 용접선이 고정이면 ㅇㅇ bottom, top 나눠서 지우는 곳 지정하면 될 듯 하네요
            if cmd == "infer":
                print(f"📩 요청 수신: rotate={rotate_angle}") # 일단 1.3으로 고정

                img_b64 = req.get("image_data", None)
                if img_b64 is None:
                    raise ValueError("image_data가 없습니다.")

                image_np = decode_base64_image(img_b64)
                if image_np is None:
                    raise ValueError("이미지 디코딩 실패")

                # max_height_ratio -> 하나의 선만 감지되었을때 original 이미지의 몇퍼센트까지 도달하면 동일한 선상에 놓여있다고 판단할것인가
                # -> 주의 해야함 오탐지가 될 수 있으니
                result = infer_image_one(model, clf, device, use_fp16, image_np, rotate_angle=rotate_angle, plot_result=CFG.plot_result)

                if isinstance(result, tuple):
                    socket.send_string(json.dumps({
                        "status": "검출 완료",
                        "upper_x": round(result[0]),
                        "lower_x": round(result[1])
                    }))
                else:
                    # 검출 실패시 -1, -1을 전송함
                    socket.send_string(json.dumps({
                        "status": "검출 실패",
                        "upper_x": result[0],
                        "lower_x": result[1]
                    }))

            else:
                socket.send_string(json.dumps({"status": "ERROR", "msg": "unknown command"}))

        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)}"
            print("❌ Error:", err_msg)
            traceback.print_exc()
            socket.send_string(json.dumps({"status": "ERROR", "msg": err_msg}))

# ==========================
# 🔹 이미지 폴더 추론 (직선 검출 포함)
# ==========================
def infer_image_one(model, clf, device, use_fp16, test_image: np.ndarray,
                    rotate_angle: float = 1.3, plot_result: bool = False):

    start_time = time.time()

    # --- 회전 적용 ---
    img_np = np.array(test_image)
    h, w = img_np.shape[:2]

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
    rotated = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # --- 상/하 분리 ---
    upper_img = rotated[:h // 2, :]
    lower_img = rotated[h // 2:, :]

    # --- upper / lower 각각 레이저 1개 검출 ---
    upper_result = detect_single_laser(model, clf, device, use_fp16, upper_img)
    lower_result = detect_single_laser(model, clf, device, use_fp16, lower_img)

    upper_x, upper_vis = upper_result  # (x좌표, 시각화 넘파이)
    lower_x, lower_vis = lower_result

    # --- Δx 계산 ---
    if upper_x < 0 or lower_x < 0:
        print("❌ 레이저 검출 실패")
        return (-1, -1)

    dx = lower_x - upper_x
    dx_text = f"{int(dx)} px"

    print(f"👉 upper_x: {upper_x}, lower_x: {lower_x}, Δx={dx_text}")
    print(f"⏱ time: {time.time() - start_time:.3f}s")

    # --- 시각화 ---
    if plot_result:
        visualize_upper_lower(rotated, upper_vis, lower_vis, upper_x, lower_x, dx_text, rotate_angle)

    return (upper_x, lower_x)

def detect_single_laser(model, clf, device, use_fp16, img_np):
    """
    입력: 상단 or 하단 이미지
    출력: (x좌표, 시각화 된 이미지)
    """

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
        feats = model.get_intermediate_layers(
            img_tensor, n=range(CFG.n_layers), reshape=True, norm=True
        )
        x = feats[-1].squeeze().detach().cpu()
        dim, H_feat, W_feat = x.shape
        x = x.view(dim, -1).permute(1, 0)

    # --- classifier ---
    fg_score = clf.predict_proba(x)[:, 1]
    fg_score = fg_score.reshape(H_feat, W_feat)
    fg_score_mf = signal.medfilt2d(fg_score, kernel_size=3)

    # --- 원본 크기로 업샘플 ---
    fg_up = cv2.resize(fg_score_mf, (w, h), interpolation=cv2.INTER_CUBIC)

    # --- threshold ---
    mask = (fg_up > 0.6).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- connected components ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean)

    # 가장 큰 area 1개만 선택
    best_label = -1
    best_area = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            best_label = label

    if best_label == -1:
        return -1, img_np  # 실패

    # --- 중심 구하기 ---
    ys, xs = np.where(labels == best_label)
    weights = fg_up[labels == best_label].astype(float)

    cx = (xs * weights).sum() / weights.sum()
    cy = (ys * weights).sum() / weights.sum()

    # --- 시각화 ---
    vis = img_np.copy()
    x0, y0, w0, h0 = stats[best_label, cv2.CC_STAT_LEFT], stats[best_label, cv2.CC_STAT_TOP], \
                     stats[best_label, cv2.CC_STAT_WIDTH], stats[best_label, cv2.CC_STAT_HEIGHT]

    cv2.rectangle(vis, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 255), 2)
    cv2.line(vis, (int(cx), y0), (int(cx), y0 + h0), (0, 0, 255), 2)
    cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)

    return round(cx), vis


# ==========================
# 🔹 유틸 함수
# ==========================
def resize_transform(img: Image, image_size: int = 768, patch_size: int = CFG.PATCH_SIZE) -> torch.Tensor:
    w, h = img.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(img, (h_patches * patch_size, w_patches * patch_size)))


def visualize_upper_lower(rotated, upper_vis, lower_vis, upper_x, lower_x, dx_text, rotate_angle):
    h, w = rotated.shape[:2]

    merged = rotated.copy()

    # upper 칸에 결과 덮기
    merged[:h//2, :] = upper_vis

    # lower 칸에 결과 덮기
    merged[h//2:, :] = lower_vis
    import matplotlib
    matplotlib.use("TkAgg")

    cv2.putText(merged, f"difference x = {dx_text}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    plt.figure(figsize=(8, 8))
    plt.title(f"rotate={rotate_angle}")
    plt.imshow(merged)
    plt.axis("off")
    plt.show(block=True)


if __name__ == "__main__":
    main()


# 이런식으로 추론 요청해야함 infer 로
# var req = new {
#     cmd = "infer",  // 👉 "추론을 해달라"는 명령 -> 나중에 infer_top_laser, infer_bottom_laser로 확장가능
#     image_data = Convert.ToBase64String(File.ReadAllBytes(imagePath)),
#     rotate_angle = 1.3
# };
# socket.Send(JsonConvert.SerializeObject(req));