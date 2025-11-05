"""
Утилита для предобработки тестового видео: сжатие до 15 FPS и обрезка по длительности.

Модуль предназначен для быстрого тестирования пайплайна на коротких видео
(до 5 минут), чтобы избежать обработки 10-часовых файлов во время разработки.

Функции:
- Снижение частоты кадров до ``target_fps`` (по умолчанию 15 FPS).
- Обрезка видео до ``max_duration_sec`` (по умолчанию 300 сек = 5 мин).
- Сохранение результата в MP4 (кодек ``mp4v``).

Путь до данного файла: test_video.py
"""
import cv2
import os
from tqdm import tqdm

from typing import Optional


def prepare_video(
    input_path: str,
    output_path: str,
    target_fps: int = 15,
    quite_duration_sec: int = 300
) -> Optional[str]:
    """
    Предобработка видео: сжатие FPS и обрезка по длительности.

    Parameters
    ----------
    input_path : str
        Путь к исходному видео (MP4/AVI).
    output_path : str
        Путь для сохранения сжатого видео.
    target_fps : int, optional
        Целевая частота кадров, по умолчанию 15.
    quite_duration_sec : int, optional
        Максимальная длительность выходного видео в секундах, по умолчанию 300 (5 минут).

    Returns
    -------
    str | None
        Путь к сохранённому видео при успехе, иначе ``None``.

    Notes
    -----
    Алгоритм:

    1. Проверка существования входного файла.
    2. Открытие видео через ``cv2.VideoCapture``.
    3. Расчёт интервала кадров: ``frame_interval = original_fps // target_fps``.
    4. Ограничение по количеству кадров: ``max_frames = quite_duration_sec * target_fps``.
    5. Чтение каждого ``frame_interval``-го кадра до достижения ``max_frames``.
    6. Сохранение результата через ``cv2.VideoWriter`` (кодек ``mp4v``).
    7. Вывод статистики и прогресс-бара через ``tqdm``.

    Examples
    --------
        python test_video.py
        # → data/output/my_video_2_15fps.mp4 (5 мин, 15 FPS)

    Важно: видео должно быть в ``data/input/``.
    """
    # 1. Проверка входного файла
    if not os.path.exists(input_path):
        print(
            f"Ошибка: Видео {input_path} не найдено.\n"
            f"Скачай тестовое видео с Pexels: "
            f"https://www.pexels.com/search/videos/body%20cam/"
        )
        return None

    # 2. Открытие видео
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видео. Проверь формат (MP4/AVI).")
        return None

    # 3. Метаданные
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        print("Ошибка: Не удалось получить FPS видео.")
        cap.release()
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / original_fps
    print(f"Оригинал: {duration_sec:.1f} сек, {original_fps:.1f} FPS, {total_frames} кадров.")

    # 4. Параметры сжатия
    frame_interval = max(1, int(original_fps / target_fps))  # Каждый N-й кадр
    max_frames = int(quite_duration_sec * target_fps)
    frames = []

    # 5. Чтение кадров с прогресс-баром
    pbar = tqdm(total=min(total_frames, max_frames), desc="Сжатие", unit="кадр")
    frame_id = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_interval == 0:
            frames.append(frame)
        frame_id += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # 6. Проверка результата
    if not frames:
        print("Предупреждение: Не удалось извлечь ни одного кадра.")
        return None

    # 7. Сохранение сжатого видео
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для .mp4
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()

    result_duration = len(frames) / target_fps
    print(
        f"Сжатое видео сохранено: {output_path}\n"
        f"  → {len(frames)} кадров, {result_duration:.1f} сек, {target_fps} FPS"
    )
    return output_path


# Тестовый запуск
if __name__ == "__main__":
    """
        Пример запуска:
            python test_video.py
        Ожидает:
            data/input/my_video_2.mp4
        Создаёт:
            data/output/my_video_2_15fps.mp4
        """
    input_video = "data/input/my_video_2.mp4"
    output_video = "data/output/my_video_2_15fps.mp4"

    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    result = prepare_video(
        input_path=input_video,
        output_path=output_video,
        target_fps=15,
        quite_duration_sec=300
    )

    if result:
        print(f"Готово! Используй в run.py: --input {result}")
    else:
        print("Не удалось обработать видео.")
