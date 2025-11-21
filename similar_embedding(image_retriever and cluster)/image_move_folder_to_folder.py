import os
import shutil
import random

def random_move_images(src_folder, dst_folder, keep_count=6200):
    """
    src_folder 안의 모든 이미지 중 랜덤으로 keep_count 개만 남기고
    나머지 파일을 dst_folder 로 이동합니다.
    """

    # 이미지 확장자
    exts = (".bmp", ".png", ".jpg", ".jpeg")

    # 목적지 폴더 없으면 생성
    os.makedirs(dst_folder, exist_ok=True)

    # 전체 파일 목록
    files = [
        f for f in os.listdir(src_folder)
        if f.lower().endswith(exts)
    ]

    total = len(files)
    print(f"총 이미지 개수: {total}")

    if total <= keep_count:
        print("⚠️ 남길 개수가 전체 개수보다 많거나 같아서 작업을 중단합니다.")
        return

    # ⭐ 랜덤으로 keep_count 만큼 샘플 선택
    keep_files = set(random.sample(files, keep_count))

    # ⭐ 나머지 파일 = 이동 대상
    move_files = [f for f in files if f not in keep_files]

    print(f"남길 이미지: {len(keep_files)}개")
    print(f"이동할 이미지: {len(move_files)}개")

    # 이동 실행
    for filename in move_files:
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)
        shutil.move(src_path, dst_path)

    print("✅ 작업 완료!")

if __name__ == "__main__":
    # random으로 keep_count 갯수만(600개) 남기고 옮깁니다.

    src = r"C:\4-2CAL_welder\dataset\Backup_현장\okng_datasets\OK\sample2"
    dst = r"C:\4-2CAL_welder\dataset\Backup_현장\okng_datasets\OK\sample3"

    random_move_images(src, dst, keep_count=5000)

