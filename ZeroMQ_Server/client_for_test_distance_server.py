import os
import zmq
import json
import base64
import re

# 🔧 처리할 이미지 폴더
folder_path = r"C:\4-2CAL_welder\dataset\Backup_현장\Backup_20251119\BOTTOM_20251118\BOTTOM_20251118012356_CSY2214(4)(0.69)_CSY6973(3)(0.80)"

# ZeroMQ 클라이언트 생성
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:55552")

# 🔍 Laser 파일만 스캔
image_files = [
    os.path.join(root, file)
    for root, _, files in os.walk(folder_path)
    for file in files
    if "Laser" in file and file.lower().endswith((".bmp", ".png", ".jpg"))
]

print(f"총 {len(image_files)}개 Laser 이미지 발견됨")

# ============================================
# 🔢 파일명에서 숫자 추출해 정렬
#     예: BOTTOM_..._Laser-001.bmp -> 1
# ============================================

def extract_number(filename):
    nums = re.findall(r'\d+', filename)
    return int(nums[-1]) if nums else -1   # 마지막 숫자 사용

# (파일경로, 숫자) 튜플로 만들기
image_files_sorted = sorted(image_files, key=lambda x: extract_number(os.path.basename(x)))

total = len(image_files_sorted)

# 🔥 앞 10장, 뒤 10장 제외
start_cut = 10
end_cut = 10

# 혹시 이미지가 20장 이하일 경우 예외 처리
if total <= (start_cut + end_cut):
    print("❗ 이미지가 20장 이하라서 제외 불가 → 모든 이미지 처리합니다.")
    target_files = image_files_sorted
else:
    target_files = image_files_sorted[start_cut : total - end_cut]

print(f"처리할 이미지 개수: {len(target_files)}개\n(앞 10장, 뒤 10장 제외됨)\n")

# ============================================
# 🔥 ZMQ 로 전송
# ============================================
for idx, image_path in enumerate(target_files, 1):

    print(f"[{idx}/{len(target_files)}] 처리 중 → {image_path}")

    # BOTTOM / TOP 구분
    if 'BOTTOM' in image_path.upper():
        rotate_angle = 2
        remove_ratio = 0.2
    else:
        rotate_angle = 1.3
        remove_ratio = 0.15

    # Base64 변환
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    # 요청 JSON
    req = {
        "cmd": "infer",
        "rotate_angle": rotate_angle,
        "remove_ratio": remove_ratio,
        "image_data": img_b64,
        "filename": os.path.basename(image_path),
    }

    socket.send_string(json.dumps(req))

    resp = json.loads(socket.recv_string())

    print("📥 응답:", resp["status"])
    print("상부 X:", resp.get("upper_x"))
    print("하부 X:", resp.get("lower_x"))
    print("-" * 50)