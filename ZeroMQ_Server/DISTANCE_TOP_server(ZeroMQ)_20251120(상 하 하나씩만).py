import zmq
import json
import traceback
import sys, os
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
    plot_result = True
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
                result = infer_image_one(model, clf, device, use_fp16, image_np, file_name="", rotate_angle=rotate_angle, remove_ratio=remove_ratio, max_height_ratio=0.7, plot_result=CFG.plot_result)

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
def infer_image_one(model, clf, device, use_fp16, test_image: np.ndarray, file_name: str = "", rotate_angle: float = 1.3, remove_ratio: float = 0.15, max_height_ratio: float = 0.7, plot_result: bool = False):
    global M_cache

    start_time = time.time()
    
    w, h = test_image.size

    # 🔹 없앨 비율 (예: 가운데 15%)
    y1 = int(h * (0.5 - remove_ratio / 2))
    y2 = int(h * (0.5 + remove_ratio / 2))

    # --- 회전 ---
    img_np = np.array(test_image)
    img_np[y1:y2, :] = 0  # 이미지 회전

    h, w = img_np.shape[:2]

    # if M_cache is None:
    #     center = (w // 2, h // 2)
    #     M_cache = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)

    # 나중에 rotation 고정이면 위에걸로 하고, 그거아닐땐 이걸로 사용해야함. -> 캐쉬때문에 디버깅이 안되서 일단 이렇게 변경
    center = (w // 2, h // 2)
    M_cache = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)    
    rotated = cv2.warpAffine(img_np, M_cache, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    test_image = Image.fromarray(rotated)

    # --- 전처리 ---
    test_image_resized = resize_transform(test_image)
    test_image_normalized = TF.normalize(test_image_resized,mean=CFG.IMAGENET_MEAN,std=CFG.IMAGENET_STD).unsqueeze(0).to(device)
    if use_fp16:
        test_image_normalized = test_image_normalized.half()

    # --- DINO 특징 ---
    with torch.inference_mode():
        feats = model.get_intermediate_layers(
            test_image_normalized, n=range(CFG.n_layers), reshape=True, norm=True
        )
        x = feats[-1].squeeze().detach().cpu()
        dim, H_feat, W_feat = x.shape
        x = x.view(dim, -1).permute(1, 0)

    # --- classifier ---
    fg_score = clf.predict_proba(x)[:, 1]
    fg_score = fg_score.reshape(H_feat, W_feat)
    fg_score_mf = signal.medfilt2d(fg_score, kernel_size=3)

    # --- 원본 크기로 업샘플 ---
    fg_score_up = cv2.resize(fg_score_mf, test_image.size, interpolation=cv2.INTER_CUBIC)

    # --- 직선 + 중심 계산 ---
    overlay_image, mask_bin, centers, detected_height = detect_laser_line(fg_score_up, np.array(test_image))

    # 중심 x 간 거리 계산
    dx_text = "N/A"

    # 여기에 centers는 2의 이하의 수임 -> detect_laser_line 함수에서 제일 스코어 값 높은 2개까지만 선정해왔음
    if (len(centers) >= 2) or (len(centers) == 1 and detected_height > h * max_height_ratio):
        # 중심점 선택
        if len(centers) >= 2:
            cx1, cy1 = centers[0]
            cx2, cy2 = centers[1]
        else:  # len == 1
            # 한개만 감지되었을때는 동일한 선상에 놓여진다고 가정-> 추후 괜찮은지 봐야함.
            cx1, cy1 = centers[0]
            cx2, cy2 = centers[0]

        # y가 더 큰 점(아래쪽 점)과 작은 점(위쪽 점) 찾기
        if cy1 > cy2:
            lower_x, lower_y = cx1, cy1
            upper_x, upper_y = cx2, cy2
        else:
            lower_x, lower_y = cx2, cy2
            upper_x, upper_y = cx1, cy1

        dx = lower_x - upper_x  # y가 큰 점의 x - y가 작은 점의 x
        dx_text = f"{int(round(dx))} px"

        # 콘솔 출력
        print(f"👉 upper_x_coordinate: {round(upper_x)}, lower_x_coordinate: {round(lower_x)}")
        print(f"✅ {file_name} | time: {time.time() - start_time:.3f}s")

        if plot_result:
            visualize_detection_result(file_name=file_name, test_image=test_image, fg_score=fg_score, fg_score_mf=fg_score_mf, overlay_image=overlay_image.copy(), upper_x=upper_x, upper_y=upper_y, lower_x=lower_x, lower_y=lower_y, cx1=cx1, cy1=cy1, cx2=cx2, cy2=cy2, dx_text=dx_text, rotate_angle=rotate_angle)

        return (upper_x, lower_x)

    else:
        print(f"감지된 laser가 2개 미만입니다.")
        return (-1, -1)

# ==========================
# 🔹 유틸 함수
# ==========================
def resize_transform(img: Image, image_size: int = 768, patch_size: int = CFG.PATCH_SIZE) -> torch.Tensor:
    w, h = img.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(img, (h_patches * patch_size, w_patches * patch_size)))

# ==========================
# 🔹 직선 검출 함수
# ==========================
def detect_laser_line(fg_score_mf: np.ndarray, original_image: np.ndarray, threshold=0.6, alpha=0.9, min_area=500):
    """
    fg_score_mf: median filter 이후 foreground heatmap (원본 크기로 업샘플 된 것)
    original_image: RGB numpy array (H, W, 3)
    alpha: 직선/박스 투명도
    min_area: 너무 작은 노이즈 컴포넌트 제거용 최소 면적
    """
    h = 0
    # 1) thresholding으로 빔 영역만 추출
    mask = (fg_score_mf > threshold).astype(np.uint8) * 255

    # 2) morphology로 노이즈 제거 및 선 강조
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3) connected components로 빔 덩어리 분리
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)

    overlay = original_image.copy()
    line_layer = np.zeros_like(overlay)

    centers = []  # (cx, cy) 리스트


    # 🔹 area 기준 상위 2개 선택
    # 🔹 y 중심 기준 upper 1개 + lower 1개 선택 (면적 top-2 대신)
    valid_labels = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        valid_labels.append(label)

    if len(valid_labels) == 0:
        return blended, mask_clean, centers, h

    # 각 라벨의 y 중심 계산
    label_y_centers = []
    for label in valid_labels:
        x0, y0, w0, h0 = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], \
                         stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        y_center = y0 + h0 / 2
        label_y_centers.append((label, y_center))

    # y 기준 정렬 (위쪽 → 아래쪽)
    label_y_centers.sort(key=lambda x: x[1])

    # upper 하나, lower 하나만 선택
    if len(label_y_centers) >= 2:
        selected_labels = [label_y_centers[0][0], label_y_centers[-1][0]]
    else:
        selected_labels = [label_y_centers[0][0]]

    # --- 선택된 라벨만 선 계산 ---
    for label in selected_labels:
        component_mask = (labels == label)
        ys, xs = np.where(component_mask)
        weights = fg_score_mf[component_mask].astype(np.float64)

        if weights.sum() == 0:
            continue

        cx = (xs * weights).sum() / weights.sum()
        cy = (ys * weights).sum() / weights.sum()
        centers.append((cx, cy))

        x, y, w, h = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], \
                    stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        cv2.rectangle(line_layer, (x, y), (x + w, y + h), (0, 255, 255), 2)
        x_int = int(round(cx))
        cv2.line(line_layer, (x_int, y), (x_int, y + h), (255, 0, 0), 2)

    # 투명도 적용
    blended = cv2.addWeighted(overlay, 1.0, line_layer, alpha, 0)

    # x 좌표 기준으로 정렬
    centers.sort(key=lambda c: c[0])

    return blended, mask_clean, centers, h


def visualize_detection_result(file_name: str, test_image, fg_score, fg_score_mf, overlay_image, upper_x, upper_y, lower_x, lower_y, cx1, cy1, cx2, cy2, dx_text: str, rotate_angle: float = 0.0):
    """
    🔍 DINOv3 추론 결과 시각화 함수
    """
    # --- 두 점 시각화 (선 + 점 + Δx 텍스트) ---
    cv2.putText(overlay_image, dx_text, (int((lower_x + upper_x) / 2), int((lower_y + upper_y) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.circle(overlay_image, (int(cx1), int(cy1)), 5, (255, 0, 0), -1) # 첫 번째 점
    cv2.circle(overlay_image, (int(cx2), int(cy2)), 5, (0, 0, 255), -1) # 두 번째 점
    cv2.line(overlay_image, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (0, 255, 0), 2)


    # --- 시각화 ---
    plt.figure(figsize=(12, 4), dpi=300)
    plt.suptitle(f"{file_name} (rotate {rotate_angle}°)  |  Δx={dx_text}")

    plt.subplot(1, 4, 1)
    plt.imshow(test_image)
    plt.axis("off")
    plt.title("Rotated Input")

    plt.subplot(1, 4, 2)
    plt.imshow(fg_score, cmap="inferno")
    plt.axis("off")
    plt.title("Foreground Score")

    plt.subplot(1, 4, 3)
    plt.imshow(fg_score_mf, cmap="inferno")
    plt.axis("off")
    plt.title("After Median Filter")

    plt.subplot(1, 4, 4)
    plt.imshow(overlay_image)
    plt.axis("off")
    plt.title("Detected Line + Δx")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

# 이런식으로 추론 요청해야함 infer 로
# var req = new {
#     cmd = "infer",  // 👉 "추론을 해달라"는 명령 -> 나중에 infer_top_laser, infer_bottom_laser로 확장가능
#     image_data = Convert.ToBase64String(File.ReadAllBytes(imagePath)),
#     rotate_angle = 1.3
# };
# socket.Send(JsonConvert.SerializeObject(req));
