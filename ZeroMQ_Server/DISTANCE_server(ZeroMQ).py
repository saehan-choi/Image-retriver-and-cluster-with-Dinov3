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
    folder_path = "D:\iba\POSCO_welding(2025.09.30~2025.12.30)\dinov3"
    model_path = "D:\iba\POSCO_welding(2025.09.30~2025.12.30)\dinov3/ibakoreaSystem/segmentation/laser_segmentation/weights/polygon_classifier_20251127.pkl"
    test_dir = rf"C:/Users/레노버/Downloads/Backup_현장/Backup_20250716/TOP_20250716142904" # 단차 차이 좀 나는거

    # 전처리 (중간 부분 제거)
    REMOVE_RATIO = 0.2

    PATCH_SIZE = 8
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    WEIGHTS_PATH = "D:\iba\POSCO_welding(2025.09.30~2025.12.30)\dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

    MODEL_TO_NUM_LAYERS = {"dinov3_vits16": 12, "dinov3_vits16plus": 12, "dinov3_vitb16": 12, "dinov3_vitl16": 24, "dinov3_vith16plus": 32, "dinov3_vit7b16": 40}
    MODEL_NAME = "dinov3_vits16"
    n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]
    IMAGE_SIZE = 768
    min_height = 60 # TOP 에서는 80이 맞고, BOTTOM에서는 width 를 조절하는게 나을거 같음
    confidence = 0.7 # activation이 해당값 이상일때만 탐지 중
    rotate_angle = 4  # TOP은 1.3도 가 제일 적절하고 , BOTTOM은 4도 width가 너무 작으면 ㅇㅇ 안하게끔 (width는 조정해볼것)
    plot_result = True if DEBUG_MODE else False
    bbox_center = True # True 시 bounding box의 정 중앙을 찾음, False시 activation의 가중 평균값을 찾음
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def write_log(message: str):
    # DEBUG 모드의 경우 print로도 띄움
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    os.makedirs(LOG_DIR, exist_ok=True)

    log_path = os.path.join(LOG_DIR, f"{today}_laser.log")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    if DEBUG_MODE:
        print(line)
        
    # log 파일 삭제는 ok_ng에서 진행함 - 365일만 남김


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
        write_log("🚀 GPU + FP16 모드로 실행 중")
    else:
        write_log("💻 CPU 모드로 실행 중 (FP32)")

    try:
        dummy = torch.randn(1, 3, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE, device=CFG.device)
        if use_fp16:
            dummy = dummy.half()
        with torch.inference_mode():
            _ = model(dummy)
        write_log("🔥 모델 warm-up 완료")
    except Exception as e:
        write_log(f"⚠️ warm-up 중 오류 발생: {e}")

    return model, clf, CFG.device, use_fp16

# 명암 구분해서 밝은곳 더 밝게 어두운 곳 더 어둡게
def s_curve(img, strength=0.5):
    """
    strength = 0.0 ~ 1.0 (0.5 추천)
    S-curve: 밝은곳↑ 어두운곳↓
    """
    img = img.astype(np.float32) / 255.0

    # S-curve
    out = img + strength * (img - img**2)  # S 커브
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return out


def main():
    model, clf, device, use_fp16 = model_load()
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")

    write_log(f"✅ ZMQ 서버 실행됨 (포트 {PORT})")

    while True:
        try:
            # 🔥 1) 멀티프레임 수신
            frames = socket.recv_multipart()

            # [Frame 1] = file name (string)
            file_name = frames[0].decode()

            # [Frame 2] = image binary
            img_bytes = frames[1]

            write_log(f"📩 요청 수신: file={file_name}")

            # 🔥 2) 이미지 디코딩
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # # 🔥 2.5) 이미지 S curve -> 해도 효과 미미함
            # image_np = s_curve(image_np, strength=0.6)

            if image_np is None:
                raise ValueError("이미지 디코딩 실패")

            # 🔥 3) 추론 수행
            result = infer_image_one(
                model, clf, device, use_fp16,
                image_np,
                rotate_angle=CFG.rotate_angle,
                plot_result=CFG.plot_result
            )

            # 🔥 4) 결과 전송(JSON만 유지)
            if isinstance(result, tuple):
                socket.send_string(json.dumps({
                    "status": "검출 완료",
                    "upper_x": round(result[0]),
                    "lower_x": round(result[1])
                }))
            else:
                socket.send_string(json.dumps({
                    "status": "검출 실패",
                    "upper_x": -1,
                    "lower_x": -1
                }))

        except Exception as e:
            err_msg = f"Error: {type(e).__name__}: {str(e)}"
            write_log(err_msg)
            traceback.print_exc()
            socket.send_string(json.dumps({"status": "ERROR", "msg": err_msg}))

# ==========================
# 🔹 이미지 폴더 추론 (직선 검출 포함)
# ==========================
def infer_image_one(model, clf, device, use_fp16, test_image,
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
    upper_result = detect_single_laser(model, clf, device, use_fp16, upper_img, min_height=CFG.min_height)
    lower_result = detect_single_laser(model, clf, device, use_fp16, lower_img, min_height=CFG.min_height)


    upper_x, upper_vis = upper_result  # (x좌표, 시각화 넘파이)
    lower_x, lower_vis = lower_result

    # --- Δx 계산 ---
    if upper_x < 0 or lower_x < 0:
        write_log("❌ 레이저 검출 실패 \n")
        return (-1, -1)

    dx = lower_x - upper_x
    dx_text = f"{int(dx)} px"

    write_log(f"👉 upper_x: {upper_x}, lower_x: {lower_x}, Δx={dx_text}")
    write_log(f"⏱ time: {time.time() - start_time:.3f}s")

    # --- 시각화 ---
    if plot_result:
        visualize_upper_lower(rotated, upper_vis, lower_vis, upper_x, lower_x, dx_text, rotate_angle)

    return (upper_x, lower_x)

def detect_single_laser(model, clf, device, use_fp16, img_np, min_height=80):
    """
    입력: 상단 or 하단 이미지
    출력: (x좌표, 시각화 된 이미지)

    min_height -> 높이가 80px 보다 작을경우 레이저 선이 아닌것으로 간주함 -> BOTTOM은 높이가 작은것도 은근 많네요
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
    mask = (fg_up > CFG.confidence).astype(np.uint8) * 255
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

    # --- 시각화 ---
    vis = img_np.copy()
    x0, y0, w0, h0 = stats[best_label, cv2.CC_STAT_LEFT], stats[best_label, cv2.CC_STAT_TOP], \
                     stats[best_label, cv2.CC_STAT_WIDTH], stats[best_label, cv2.CC_STAT_HEIGHT]

    # --- 중심 구하기 ---
    if CFG.bbox_center:
        # --- 중심 구하기: Bounding Box의 기하학적 중앙 ---
        cx = x0 + w0 / 2
        cy = y0 + h0 / 2
    else:
        ys, xs = np.where(labels == best_label)
        weights = fg_up[labels == best_label].astype(float)

        cx = (xs * weights).sum() / weights.sum()
        cy = (ys * weights).sum() / weights.sum()

    if h0 < min_height:
        # 이게 짧으면 오탐이 뜨고, 길면 실제 탐지되야 할 것 도 안됨.
        write_log(f'탐지된 박스의 높이가 최소 높이보다 작은 상태입니다. 예측 높이: {h0} / 최소 높이: {min_height}')
        return -1, img_np

    cv2.rectangle(vis, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 255), 2)
    cv2.line(vis, (int(cx), y0), (int(cx), y0 + h0), (0, 0, 255), 2)
    cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)

    # ============================
    # 🔥 DEBUG 모드일 때만 heatmap 계산
    # ============================
    if CFG.plot_result:
        heatmap = (fg_up * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 원본 vis + heatmap overlay
        overlay = cv2.addWeighted(vis, 0.85, heatmap_color, 0.15, 0)

        return round(cx), overlay

    # DEBUG off → heatmap 없이 기본 bbox만 반환
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

    merged_overlay = rotated.copy()
    merged_overlay[:h//2, :] = upper_vis
    merged_overlay[h//2:, :] = lower_vis

    # RGB 변환
    merged_overlay = cv2.cvtColor(merged_overlay, cv2.COLOR_BGR2RGB)
    rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

    import matplotlib
    matplotlib.use("TkAgg")

    fig = plt.figure(figsize=(12, 6))

    # 🔥 창 위치 고정 : 가로 중앙 + 세로 최상단
    manager = plt.get_current_fig_manager()
    window = manager.window
    window.update_idletasks()

    x = 300
    y = 0
    window.geometry(f"+{x}+{y}")

    # --- subplot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(rotated_rgb)
    ax1.set_title(f"① 회전된 이미지 (rotate={rotate_angle}°)")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(merged_overlay)
    ax2.set_title(f"② 레이저 검출 Overlay (Δx={dx_text})")
    ax2.axis("off")

    # --- Colorbar ---
    cmap = matplotlib.colormaps.get_cmap("jet")
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("Activation (0 ~ 1)")

    cb.ax.hlines(f'{CFG.confidence}', 0, 1, colors="black", linewidth=2)
    cb.ax.text(1.1, CFG.confidence, f'{CFG.confidence}', color="black", va="center", fontsize=8)

    plt.show(block=True)


if __name__ == "__main__":
    main()