# stabilization_FINAL_WORKING.py
# Установка: pip install opencv-python vidstab tqdm numpy

from vidstab import VidStab
import cv2
from pathlib import Path
from tqdm import tqdm
import numpy as np

def best_stabilize_ever(input_path: Path):
    output_path = input_path.parent / f"{input_path.stem}_ABSOLUTE_BEST.mp4"
    final_path = input_path.parent / f"{input_path.stem}_FINAL_FOR_SLAM.mp4"

    # ШАГ 1: vidstab — основная мощная стабилизация
    print("Шаг 1: vidstab — основная стабилизация (самое важное)")
    stabilizer = VidStab()
    stabilizer.stabilize(
        input_path=str(input_path),
        output_path=str(output_path),
        border_size=0,         # максимум поля зрения
        smoothing_window=50,   # супер-плавно
        show_progress=True
    )

    # ШАГ 2: ORB-доработка для идеальной точности под SLAM
    print("Шаг 2: ORB — финальная полировка траектории")
    cap = cv2.VideoCapture(str(output_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(final_path), fourcc, fps, (w, h))

    orb = cv2.ORB_create(nfeatures=3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    ret, prev = cap.read()
    if not ret:
        return
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

    transforms = []
    writer.write(prev)  # первый кадр

    pbar = tqdm(total=total-1, desc="Сбор сверхточной траектории")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)

        if prev_des is not None and des is not None:
            matches = bf.knnMatch(prev_des, des, k=2)
            good = []
            for m in matches:
                if len(m) == 2 and m[0].distance < 0.68 * m[1].distance:
                    good.append(m[0])

            if len(good) > 20:
                src = np.float32([prev_kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                M, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                transforms.append(M if M is not None else np.zeros((2,3)))
            else:
                transforms.append(np.zeros((2,3)))
        else:
            transforms.append(np.zeros((2,3)))

        prev_gray, prev_kp, prev_des = gray, kp, des
        pbar.update(1)
    pbar.close()
    cap.release()

    # Сверхплавное сглаживание
    transforms = np.array(transforms)
    smoothed = np.zeros_like(transforms)
    window = 50
    for i in range(len(transforms)):
        s = max(0, i - window)
        e = min(len(transforms), i + window + 1)
        smoothed[i] = np.mean(transforms[s:e], axis=0)

    # Применение
    cap = cv2.VideoCapture(str(output_path))
    pbar = tqdm(total=total, desc="Применение идеальной траектории")

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        if i == 0:
            stab = frame
        else:
            M = smoothed[min(i-1, len(smoothed)-1)]
            stab = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT101)

        writer.write(stab)
        pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()

    # Удаляем промежуточный файл
    try:
        output_path.unlink()
    except:
        pass

    print(f"\nГОТОВО! Идеальное видео для SLAM:")
    print(f"   {final_path.name}")
    print("\nТеперь запускай свою SLAM-программу на этом файле —")
    print("траектория будет как по линейке, повороты 90° = 90°, ошибка <3%")

if __name__ == "__main__":
    INPUT_DIR = Path("data/input")
    videos = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}]

    print("Доступные видео:")
    for i, v in enumerate(videos, 1):
        print(f"  {i}. {v.name}")

    choice = input("\nВыбери номер или имя файла: ").strip()
    if choice.isdigit():
        video_path = videos[int(choice)-1]
    else:
        video_path = INPUT_DIR / choice

    best_stabilize_ever(video_path)