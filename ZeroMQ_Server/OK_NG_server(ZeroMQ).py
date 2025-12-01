import os, zmq, json, cv2, numpy as np, base64, datetime, traceback
import torch

import albumentations as A

from torch import nn
from albumentations.pytorch import ToTensorV2
from concurrent.futures import ThreadPoolExecutor

# ==========================
# ⚙️ 설정
# ==========================
DEBUG_MODE = True
LOG_DIR = "logs"
PORT = 55551


# ==========================
# 🧾 로그 함수
# ==========================
def write_log(message: str):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{today}_okng.log")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    if DEBUG_MODE:
        print(line)

    cleanup_logs(keep_last_n=365) # 최근 365개 파일만 남기고 다른건 삭제

def cleanup_logs(keep_last_n=5):
    """
    LOG_DIR 안의 로그 파일 중 최신 N개만 남기고 오래된 파일 삭제
    삭제 내용도 오늘 로그 파일에 안전하게 기록 (무한루프 없음)
    """
    if not os.path.exists(LOG_DIR):
        return

    log_files = sorted(
        [os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR) if f.endswith(".log")],
        key=os.path.getmtime
    )

    if len(log_files) <= keep_last_n:
        return

    files_to_delete = log_files[:-keep_last_n]

    # 오늘 로그 파일에 삭제 기록 남겨야 하니까 경로 준비
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    today_log_path = os.path.join(LOG_DIR, f"{today}_okng.log")

    for f in files_to_delete:
        try:
            os.remove(f)

            # 🔥 write_log() 호출 금지 → 무한루프 막기
            # 대신 직접 파일에 한 줄만 append
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{timestamp}] 🗑 오래된 로그 삭제: {os.path.basename(f)}\n"

            with open(today_log_path, "a", encoding="utf-8") as logf:
                logf.write(line)

            if DEBUG_MODE:
                print(line.strip())

        except Exception as e:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{timestamp}] ❌ 로그 삭제 실패: {e}\n"

            with open(today_log_path, "a", encoding="utf-8") as logf:
                logf.write(line)

            if DEBUG_MODE:
                print(line.strip())

# ==========================
# 🧠 모델 설정
# ==========================
class CFG:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dinov3_location = rf"D:\iba\POSCO_welding(2025.09.30~2025.12.30)\dinov3/"
    model_name = "dinov3_vits16"
    dinov3_weights_path = rf"D:\iba\POSCO_welding(2025.09.30~2025.12.30)\dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    weights_path = rf"D:\iba\POSCO_welding(2025.09.30~2025.12.30)\dinov3\ibakoreaSystem\classification\weights\DINOv3_linear_best_epoch50.pth"
    in_features = 384
    model_num_class = 2
    batch_size = 8   # ✅ 기본 배치 크기 (C#에서 덮어쓸 수도 있음)
    img_resize = (418, 418)
    skip_img_cnt_head = 4
    skip_img_cnt_tail = 4
    use_fp16 = True


# ==========================
# 🧠 DINOv3 모델
# ==========================
class DinoLinearClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = torch.hub.load(
            repo_or_dir=CFG.dinov3_location,
            model=CFG.model_name,
            source="local",
            weights=CFG.dinov3_weights_path,
        )
        self.fc = nn.Linear(CFG.in_features, num_classes)

    def forward(self, x):
        with torch.no_grad():
            outputs = self.model.forward_features(x)
            if isinstance(outputs, dict):
                x = outputs.get("x_norm_clstoken", list(outputs.values())[0])
            else:
                x = outputs
        return self.fc(x)


# ==========================
# 🚀 모델 초기화
# ==========================
model = None

def init_model():
    global model
    try:
        model = DinoLinearClassifier(num_classes=CFG.model_num_class).to(CFG.device)
        state_dict = torch.load(CFG.weights_path, map_location=CFG.device)
        fixed_state_dict = {k.replace("backbone.", "model."): v for k, v in state_dict.items()}
        model.load_state_dict(fixed_state_dict, strict=False)
        model.eval()

        if CFG.use_fp16 and CFG.device == "cuda":
            model.half()
            write_log("🚀 GPU + FP16 모드로 실행 중")
        else:
            write_log("💻 CPU 모드로 실행 중 (FP32)")

        # 더미 테스트
        dummy = torch.randn(1, 3, *CFG.img_resize).to(CFG.device)
        if CFG.use_fp16 and CFG.device == "cuda":
            dummy = dummy.half()
        outputs = model(dummy)
        write_log("🔥 모델 warm-up 완료")
        return True
    except Exception as e:
        write_log(f"❌ 모델 초기화 실패: {e}")
        traceback.print_exc()
        return False


# ==========================
# 🧩 이미지 변환 함수
# ==========================
def decode_base64_image(img_b64):
    try:
        img_bytes = base64.b64decode(img_b64)
        img_np = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    except Exception as e:
        write_log(f"❌ 이미지 디코딩 실패: {e}")
        return None


# ==========================
# 🔮 배치 추론
# ==========================
def execute_inference_batch(frames):
    try:

        write_log(f"📦 Batch 요청 수신: {len(frames)}장")

        images = []
        for raw_bytes in frames:
            img_np = np.frombuffer(raw_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)

        transform = A.Compose([
            A.Resize(*CFG.img_resize),
            A.Normalize(),
            ToTensorV2(),
        ])

        # 🔹 ThreadPoolExecutor로 병렬 변환
        with ThreadPoolExecutor(max_workers=8) as executor:
            tensors = list(executor.map(
                lambda img: transform(
                    image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                )["image"],
                images
            ))

        tensors = [t for t in tensors if t is not None]
        if not tensors:
            raise ValueError("유효한 이미지가 없습니다.")

        # 🔹 배치 텐서로 합치기
        batch_tensor = torch.stack(tensors).to(CFG.device)
        if CFG.use_fp16 and CFG.device == "cuda":
            batch_tensor = batch_tensor.half()
            model.half()

        # 🔹 추론
        model.eval()
        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_classes = torch.argmax(probs, dim=1).cpu().numpy()
            confidences = probs.max(dim=1).values.cpu().numpy()

        # 🔹 결과 생성
        results = []
        for idx, (cls, conf) in enumerate(zip(pred_classes, confidences)):
            label = "OK" if cls == 0 else "NG"
            results.append({
                "index": idx,
                "result": label,
                "confidence": round(float(conf), 3),
                "class_id": int(cls)
            })

        write_log(f"✅ Batch 추론 완료 — {len(results)}개 처리됨.")
        return results

    except Exception as e:
        write_log(f"❌ Batch 추론 중 오류: {e}")
        traceback.print_exc()
        return [{"status": "ERROR", "msg": str(e)}]


# ==========================
# 🌐 ZMQ 서버
# ==========================
def run_server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")
    init_model()

    write_log(f"✅ ZMQ 서버 실행됨 (포트 {PORT})")

    while True:
        try:
            # ✅ 여러 프레임(이미지) 수신 
            frames = socket.recv_multipart()  # 여러 프레임 받기
            write_log(f"📩 받은 프레임 수: {len(frames)}")

            # 요청 타입에 따라 단일 / 배치 구분
            result = execute_inference_batch(frames)

            socket.send_string(json.dumps(result, ensure_ascii=False))

        except Exception as e:
            err_msg = f"❌ 서버 오류: {e}"
            write_log(err_msg)
            socket.send_string(json.dumps({"status": "ERROR", "msg": str(e)}))
            traceback.print_exc()


# ==========================
# 🧩 실행
# ==========================
if __name__ == "__main__":
    run_server()

