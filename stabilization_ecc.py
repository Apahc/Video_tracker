"""
ECC-стабилизация видео (OpenCV).
- Полная устойчивость к ошибкам
- Прогресс-бары
- mp4v
- БЕЗ ПАДЕНИЙ
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from scipy.fft import rfft, rfftfreq


def smooth_trajectory(
    transforms: np.ndarray,
    window_size: int = 30
) -> np.ndarray:
    """
    Сглаживает траекторию с помощью скользящего среднего.
    """
    smoothed = np.zeros_like(transforms)
    for ind in range(transforms.shape[0]):
        start = max(0, ind - window_size // 2)
        end = min(transforms.shape[0], ind + window_size // 2 + 1)
        smoothed[ind] = np.mean(transforms[start:end], axis=0)
    return smoothed


def auto_smoothing_window(
    transforms: np.ndarray,
    fps: float,
    motion_type: int = cv2.MOTION_AFFINE
) -> int:
    """
    Автоподбор smoothing_window по частоте шагов (FFT).

    Parameters
    ----------
    transforms : np.ndarray
        Массив матриц движения.
    fps : float
        FPS видео.
    motion_type : int
        Тип движения.

    Returns
    -------
    int
        Оптимальный smoothing_window.
    """
    if len(transforms) < 30:
        return 45  # fallback

    # Извлекаем dx, dy
    dx = []
    dy = []
    for M in transforms:
        if motion_type == cv2.MOTION_HOMOGRAPHY:
            dx.append(M[0, 2])
            dy.append(M[1, 2])
        else:
            dx.append(M[0, 2])
            dy.append(M[1, 2])
    dx = np.array(dx)
    dy = np.array(dy)

    # Среднее по осям
    signal = (dx + dy) / 2
    signal = signal - np.mean(signal)  # убираем дрейф

    # FFT
    N = len(signal)
    yf = rfft(signal)           # Правильно, быстро, для вещественного сигнала
    xf = rfftfreq(N, 1 / fps)   # Правильно, даёт частоты в Гц

    # Находим пик в диапазоне 0.5–3 Гц (шаг человека)
    mask = (xf >= 0.5) & (xf <= 3.0)
    if not np.any(mask):
        return 45

    freqs = xf[mask]
    amps = np.abs(yf[mask])
    if len(amps) == 0:
        return 45

    peak_freq = freqs[np.argmax(amps)]

    if peak_freq <= 0:
        return 45

    # smoothing_window = 2 / freq (в кадрах)
    window = int(2 * fps / peak_freq)
    window = max(15, min(window, 120))  # ограничиваем

    return window


def stabilize_video_ecc(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    motion_type: int = cv2.MOTION_AFFINE,
    smoothing_window: int = 45,
    border_size: int = 120,
    make_overlay: bool = True,
    make_plots: bool = True,
    max_iterations: int = 200,
    epsilon: float = 1e-10
) -> Dict[str, Path]:
    """
    Стабилизация видео с помощью ECC.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {input_path}")

    if output_dir is None:
        output_dir = input_path.parent / "ecc_stab"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    stem = input_path.stem
    suffix = input_path.suffix
    stabilized_path = output_dir / f"{stem}_ecc_stabilize{suffix}"
    overlay_path = output_dir / f"{stem}_ecc_overlay{suffix}" if make_overlay else None
    traj_plot_path = output_dir / f"{stem}_trajectory.png" if make_plots else None

    print(f"ECC-стабилизация: {input_path.name}")
    print(f"Результат → {stabilized_path.name}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError("Не открыть видео")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(stabilized_path), fourcc, fps, (w, h))

    # Первый кадр
    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Нет кадров")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Инициализация
    if motion_type == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    transforms = []
    pbar = tqdm(total=total_frames, desc="Анализ ECC", unit="кадр")

    writer.write(prev_frame)
    pbar.update(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        input_matrix = warp_matrix.copy()

        try:
            result = cv2.findTransformECC(
                prev_gray, gray, input_matrix,
                motion_type,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon)
            )

            # Автоопределение порядка
            if isinstance(result, tuple) and len(result) == 2:
                if isinstance(result[0], (float, np.floating)):
                    correlation, new_matrix = result
                else:
                    new_matrix, correlation = result
            else:
                correlation = 0.0
                new_matrix = input_matrix

            if correlation > 0.6:
                warp_matrix = new_matrix

        except cv2.error as e:
            print(f"ECC ошибка: {e} — используем предыдущую матрицу")

        transforms.append(warp_matrix.copy())
        prev_gray = gray
        pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()

    # === Сглаживание1 ===
    # if transforms:
    #     transforms_array = np.array(transforms)
    #     smoothed = smooth_trajectory(transforms_array, smoothing_window)
    # else:
    #     if motion_type == cv2.MOTION_HOMOGRAPHY:
    #         smoothed = np.array([np.eye(3, 3, dtype=np.float32)])
    #     else:
    #         smoothed = np.array([np.eye(2, 3, dtype=np.float32)])

    # === Сглаживание2 ===
    if transforms:
        transforms_array = np.array(transforms)

        # === АВТОПОДБОР ===
        try:
            from scipy.fft import rfft, rfftfreq
            auto_window = auto_smoothing_window(transforms_array, fps, motion_type)
            print(f"Автоподбор: smoothing_window = {auto_window}")
            smoothing_window = auto_window
        except Exception as e:
            print(f"Автоподбор не удался: {e}. Используем {smoothing_window}")

        smoothed = smooth_trajectory(transforms_array, smoothing_window)
    else:
        if motion_type == cv2.MOTION_HOMOGRAPHY:
            smoothed = np.array([np.eye(3, 3, dtype=np.float32)])
        else:
            smoothed = np.array([np.eye(2, 3, dtype=np.float32)])

    # === Применение ===
    cap = cv2.VideoCapture(str(input_path))
    writer = cv2.VideoWriter(str(stabilized_path), fourcc, fps, (w, h))

    pbar = tqdm(total=total_frames, desc="Применение ECC", unit="кадр")

    for ind in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        M = smoothed[min(ind, len(smoothed) - 1)]

        if motion_type == cv2.MOTION_HOMOGRAPHY:
            stabilized = cv2.warpPerspective(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        else:
            stabilized = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Кроп
        crop = border_size
        if crop > 0 and crop * 2 < min(w, h):
            stabilized = stabilized[crop:-crop, crop:-crop]
            stabilized = cv2.resize(stabilized, (w, h), interpolation=cv2.INTER_LINEAR)

        writer.write(stabilized)
        pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()

    stab_results: Dict[str, Path] = {"stabilized": stabilized_path}

    # === Overlay ===
    if make_overlay and overlay_path:
        create_overlay_video(input_path, stabilized_path, overlay_path)
        stab_results["overlay"] = overlay_path

    # === График ===
    if make_plots and transforms:
        import matplotlib.pyplot as plt
        traj = []
        for M in transforms:
            if motion_type == cv2.MOTION_HOMOGRAPHY:
                dx = M[0, 2]
                dy = M[1, 2]
                angle = np.arctan2(M[1, 0], M[0, 0])
            else:
                dx = M[0, 2]
                dy = M[1, 2]
                angle = np.arctan2(M[1, 0], M[0, 0])
            traj.append([dx, dy, angle])
        traj = np.array(traj)

        plt.figure(figsize=(10, 6))
        plt.plot(traj[:, 0], label="dx", alpha=0.8)
        plt.plot(traj[:, 1], label="dy", alpha=0.8)
        plt.plot(np.degrees(traj[:, 2]), label="угол", alpha=0.8)
        plt.legend()
        plt.title("ECC: Траектория камеры")
        plt.xlabel("Кадр")
        plt.ylabel("Смещение")
        plt.grid(True, alpha=0.3)
        if traj_plot_path:
            plt.savefig(str(traj_plot_path), dpi=150, bbox_inches='tight')
            plt.close()
            stab_results["trajectory_plot"] = traj_plot_path

    print("Готово!\n")
    return stab_results


def create_overlay_video(
    original_path: Path,
    stabilized_path: Path,
    output_path: Path
) -> None:
    """
    Создаёт 50/50 overlay.
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
    OUTPUT_DIR = INPUT_DIR / "ecc_stab"

    videos = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
    if not videos:
        raise FileNotFoundError("Нет видео в data/input/")

    print("Доступные видео:")
    for i, v in enumerate(videos, 1):
        print(f"  {i}. {v.name}")

    while True:
        choice = input(f"\nВведите номер (1–{len(videos)}) или имя: ").strip()
        if not choice:
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(videos):
            video_path = videos[int(choice) - 1]
            break
        candidate = INPUT_DIR / choice
        if candidate in videos:
            video_path = candidate
            break
        print("Неверно.")

    results = stabilize_video_ecc(
        input_path=video_path,
        output_dir=OUTPUT_DIR,
        smoothing_window=60,
        border_size=150,
        motion_type=cv2.MOTION_AFFINE,
        make_overlay=True,
        make_plots=True
    )

    print("Создано:")
    for k, v in results.items():
        print(f"  • {k:18}: {v.name}")