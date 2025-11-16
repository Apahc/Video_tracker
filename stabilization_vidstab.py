"""
Модуль для стабилизации видео с bodycam с использованием библиотеки ``vidstab``.

Функционал:
    - Стабилизация видео по ключевым точкам (ORB/GFTT)
    - Сохранение результатов в подпапку ``vidstab``
    - Создание overlay-видео (оригинал ↔ стабилизированное)
    - Сохранение графиков траектории и трансформаций
    - Прогресс-бар для overlay
    - Кодек ``mp4v`` — совместим с OpenCV

!pip install vidstab
!pip install tqdm
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm

from vidstab import VidStab


def create_overlay_video(
    original_path: Path,
    stabilized_path: Path,
    output_path: Path
) -> None:
    """
    Создаёт видео с наложением: левая половина — оригинал, правая — стабилизированное.

    Parameters
    ----------
    original_path : Path
        Путь к исходному видео.
    stabilized_path : Path
        Путь к стабилизированному видео.
    output_path : Path
        Путь для сохранения overlay-видео (формат ``mp4``, кодек ``mp4v``).

    Returns
    -------
    None
        Видео сохраняется на диск.
    """
    cap_orig = cv2.VideoCapture(str(original_path))
    cap_stab = cv2.VideoCapture(str(stabilized_path))

    if not cap_orig.isOpened() or not cap_stab.isOpened():
        raise RuntimeError("Не удалось открыть одно из видео для создания overlay")

    w = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    print(f"Создание overlay → {output_path.name}...")

    pbar = tqdm(total=total_frames, desc="Overlay", unit="кадр", leave=False)

    while True:
        ret1, frame1 = cap_orig.read()
        ret2, frame2 = cap_stab.read()
        if not ret1 or not ret2:
            break

        frame2 = cv2.resize(frame2, (w, h), interpolation=cv2.INTER_LINEAR)
        mid = w // 2

        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[:, :mid] = frame1[:, :mid]
        overlay[:, mid:] = frame2[:, mid:]

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


def save_plot_from_vidstab(
    plot_func,
    output_path: Path,
    title: str
) -> bool:
    """
    Сохраняет график, возвращаемый функцией ``vidstab`` (например, ``plot_trajectory``).

    Parameters
    ----------
    plot_func : callable
        Функция, возвращающая ``(fig, ax)`` или ``fig``.
    output_path : Path
        Путь для сохранения PNG.
    title : str
        Название для логов (например, "траектория").

    Returns
    -------
    bool
        ``True`` — если график сохранён успешно.

    Notes
    -----
    - Обрабатывает как ``(fig, ax)``, так и ``fig``.
    - Закрывает фигуру после сохранения.
    """
    try:
        result = plot_func()
        if result is None:
            return False
        fig = result[0] if isinstance(result, tuple) and len(result) >= 1 else result
        if fig is None:
            return False
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Ошибка при сохранении графика '{title}': {e}")
        return False


def stabilize_video_with_vidstab(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    kp_method: str = "ORB",
    smoothing_window: int = 45,
    border_type: str = "reflect",
    border_size: int = 120,
    make_overlay: bool = True,
    make_plots: bool = True
) -> Dict[str, Path]:
    """
    Выполняет стабилизацию видео с помощью ``vidstab`` и сохраняет все артефакты.

    Parameters
    ----------
    input_path : str or Path
        Путь к исходному видео.
    output_dir : str or Path, optional
        Папка для результатов. Если ``None`` — создаётся ``vidstab`` в родительской папке.
    kp_method : str, default ``"ORB"``
        Метод детекции ключевых точек (``"ORB"``, ``"GFTT"``).
    smoothing_window : int, default ``45``
        Размер окна сглаживания траектории. Больше — плавнее.
    border_type : str, default ``"reflect"``
        Тип заполнения краёв (``"reflect"``, ``"replicate"``).
    border_size : int, default ``120``
        Запас пикселей под кроп.
    make_overlay : bool, default ``True``
        Создавать ли сравнительное видео.
    make_plots : bool, default ``True``
        Сохранять ли графики траектории и трансформаций.

    Returns
    -------
    dict[str, Path]
        Словарь с путями к файлам:
        - ``"stabilized"`` — стабилизированное видео
        - ``"overlay"`` — (если создано) сравнительное видео
        - ``"trajectory_plot"`` — (если создано) график траектории
        - ``"transforms_plot"`` — (если создано) график трансформаций

    Examples
    --------
    stabilize_video_with_vidstab("data/input/run.mp4", smoothing_window=60)
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {input_path}")

    if output_dir is None:
        output_dir = input_path.parent / "vidstab"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    stabilizer = VidStab(kp_method=kp_method)

    stem = input_path.stem
    suffix = input_path.suffix
    stabilized_path = output_dir / f"{stem}_vidstab_stabilize{suffix}"
    overlay_path = output_dir / f"{stem}_vidstab_overlay{suffix}" if make_overlay else None
    traj_plot_path = output_dir / f"{stem}_trajectory.png" if make_plots else None
    trans_plot_path = output_dir / f"{stem}_transforms.png" if make_plots else None

    print(f"Стабилизация: {input_path.name}")
    print(f"Результат → {stabilized_path.name}...")

    # Стабилизация
    stabilizer.stabilize(
        input_path=str(input_path),
        output_path=str(stabilized_path),
        smoothing_window=smoothing_window,
        border_type=border_type,
        border_size=border_size,
        show_progress=False
    )

    stab_results: Dict[str, Path] = {"stabilized": stabilized_path}

    # Overlay
    if make_overlay and overlay_path:
        try:
            create_overlay_video(input_path, stabilized_path, overlay_path)
            stab_results["overlay"] = overlay_path
        except Exception as e:
            print(f"Ошибка при создании overlay: {e}")

    # Графики
    if make_plots:
        print("Сохраняю графики...")
        if traj_plot_path:
            save_plot_from_vidstab(stabilizer.plot_trajectory, traj_plot_path, "траектория")
        if trans_plot_path:
            save_plot_from_vidstab(stabilizer.plot_transforms, trans_plot_path, "трансформации")

    print("Готово!\n")
    return stab_results


# ======================= ОСНОВНОЙ БЛОК =======================
if __name__ == "__main__":
    INPUT_DIR = Path("data/input")
    VIDSTAB_DIR = INPUT_DIR / "vidstab"

    VIDEO_NAME: Optional[str] = None

    if not VIDEO_NAME:
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

    else:
        video_path = INPUT_DIR / VIDEO_NAME
        if not video_path.exists():
            raise FileNotFoundError(f"Файл не найден: {video_path}")

    # Запуск
    results = stabilize_video_with_vidstab(
        input_path=video_path,
        output_dir=VIDSTAB_DIR,
        kp_method="ORB",
        smoothing_window=45,
        border_size=130,
        make_overlay=True,
        make_plots=True
    )

    print("Созданные файлы:")
    for key, path in results.items():
        print(f"  • {key:18}: {path.name}")
