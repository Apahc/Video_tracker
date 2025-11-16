"""
Гибридная стабилизация: vidstab → OpenCV
- Быстро + точно
- Для bodycam
"""

from vidstab import VidStab
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from typing import Dict


def hybrid_stabilize(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    smoothing_window: int = 45,
    border_size: int = 120,
    make_overlay: bool = True
) -> Dict[str, Path]:
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent / "hybrid_stab"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # === ШАГ 1: vidstab (грубая) ===
    print("Шаг 1: vidstab — грубая стабилизация")
    stabilizer = VidStab()
    rough_path = output_dir / f"{input_path.stem}_vidstab_rough.mp4"
    stabilizer.stabilize(
        input_path=str(input_path),
        output_path=str(rough_path),
        border_size=border_size,
        smoothing_window=30,  # грубое сглаживание
        show_progress=True
    )

    # === ШАГ 2: OpenCV (точная доработка) ===
    print("Шаг 2: OpenCV — точная доработка")
    final_path = output_dir / f"{input_path.stem}_final_stabilize.mp4"
    overlay_path = output_dir / f"{input_path.stem}_final_overlay.mp4" if make_overlay else None

    stabilize_orb_refine(rough_path, final_path, smoothing_window, border_size)

    # === Overlay (по желанию) ===
    results: Dict[str, Path] = {
        "rough": rough_path,
        "final": final_path
    }
    if make_overlay and overlay_path:
        create_overlay_video(input_path, final_path, overlay_path)
        results["overlay"] = overlay_path

    return results


# === OpenCV-дообработка (ORB) ===
def stabilize_orb_refine(
    input_path: Path,
    output_path: Path,
    smoothing_window: int,
    border_size: int
) -> None:
    cap = cv2.VideoCapture(str(input_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    orb = cv2.ORB_create(500)   # type: ignore
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

    transforms = []
    pbar = tqdm(total=total, desc="OpenCV доработка")

    writer.write(prev)
    pbar.update(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if des is None:
            writer.write(frame)
            pbar.update(1)
            continue

        matches = matcher.knnMatch(prev_des, des, k=2)
        good = []
        for match in matches:
            if len(match) == 2:  # Убедимся, что есть 2 совпадения
                m, n = match
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            else:
                # Если только 1 совпадение — пропускаем (или можно взять m)
                pass

        if len(good) > 10:
            src = np.float32([prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, _ = cv2.estimateAffine2D(src, dst)
            if M is not None:
                transforms.append(M)
            else:
                transforms.append(np.eye(3)[:2])
        else:
            transforms.append(np.eye(3)[:2])

        prev_gray = gray
        prev_kp, prev_des = kp, des
        pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()

    # Сглаживание
    if transforms:
        transforms = np.array(transforms)
        smoothed = smooth_trajectory(transforms, smoothing_window)

    # Применение
    cap = cv2.VideoCapture(str(input_path))
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    pbar = tqdm(total=total, desc="Применение доработки")

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        M = smoothed[min(i, len(smoothed)-1)]
        stab = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if border_size > 0:
            stab = stab[border_size:-border_size, border_size:-border_size]
            stab = cv2.resize(stab, (w, h))
        writer.write(stab)
        pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()


# === smooth_trajectory (как у тебя) ===
def smooth_trajectory(transforms, window):
    smoothed = np.zeros_like(transforms)
    for i in range(len(transforms)):
        s = max(0, i - window//2)
        e = min(len(transforms), i + window//2 + 1)
        smoothed[i] = np.mean(transforms[s:e], axis=0)
    return smoothed


def create_overlay_video(
    original_path: Path,
    stabilized_path: Path,
    output_path: Path
) -> None:
    """
    Создаёт 50/50 overlay: оригинал (лево) ↔ стабилизированное (право).

    Parameters
    ----------
    original_path : Path
        Путь к оригиналу.
    stabilized_path : Path
        Путь к стабилизированному.
    output_path : Path
        Путь для overlay.

    Examples
    --------
    create_overlay_video(
        Path("in.mp4"),
        Path("out_stab.mp4"),
        Path("overlay.mp4")
    )
    """
    cap_orig = cv2.VideoCapture(str(original_path))
    cap_stab = cv2.VideoCapture(str(stabilized_path))

    if not cap_orig.isOpened() or not cap_stab.isOpened():
        raise RuntimeError("Не удалось открыть видео для overlay")

    w = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    total = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    print(f"Создаю overlay → {output_path.name}")
    pbar = tqdm(total=total, desc="Overlay", unit="кадр", leave=False)

    while True:
        r1, f1 = cap_orig.read()
        r2, f2 = cap_stab.read()
        if not r1 or not r2:
            break
        f2 = cv2.resize(f2, (w, h), interpolation=cv2.INTER_LINEAR)
        mid = w // 2
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[:, :mid] = f1[:, :mid]
        overlay[:, mid:] = f2[:, mid:]
        cv2.line(overlay, (mid, 0), (mid, h), (0, 255, 0), 2)
        cv2.putText(overlay, "ORIGINAL", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(overlay, "STABILIZED", (mid + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        writer.write(overlay)
        pbar.update(1)

    pbar.close()
    cap_orig.release()
    cap_stab.release()
    writer.release()
    print("Overlay готов")


# ======================= ОСНОВНОЙ БЛОК =======================
if __name__ == "__main__":
    INPUT_DIR = Path("data/input")
    OUTPUT_DIR = INPUT_DIR / "hybrid_stab"

    videos = [
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    ]
    if not videos:
        raise FileNotFoundError("Нет видео в data/input/")

    print("Доступные видео:\n")
    for i, v in enumerate(videos, 1):
        print(f"  {i}. {v.name}")

    while True:
        choice = input(f"\nВведите номер (1–{len(videos)}) или имя файла: ").strip()
        if not choice:
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(videos):
            video_path = videos[int(choice) - 1]
            break
        candidate = INPUT_DIR / choice
        if candidate in videos:
            video_path = candidate
            break
        print("Неверный ввод.")

    # === Запуск гибридной стабилизации ===
    results = hybrid_stabilize(
        input_path=video_path,
        output_dir=OUTPUT_DIR,
        smoothing_window=60,
        border_size=150
    )

    print("\nСозданные файлы:")
    print(f"  • rough_vidstab   : {Path(results['rough']).name}")
    print(f"  • final_stabilized: {Path(results['final']).name}")
    if "overlay" in results:
        print(f"  • overlay         : {Path(results['overlay']).name}")