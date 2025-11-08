"""
Основной пайплайн обработки видео для оффлайн-анализа траектории движения.

Модуль реализует полный цикл обработки одного видеофайла:
1. Чтение кадров с видеорегистратора.
2. Оценка глубины с помощью MiDaS (масштабирование в метры).
3. Трекинг камеры и построение траектории с помощью ORB-SLAM3 (упрощённая версия).
4. Расчёт километража, средней скорости и статистики.
5. Сохранение результата в JSON-файл.

Путь до данного файла: src/pipeline.py
"""
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from src.midas import MiDaSEstimator
from src.slam import ORBSLAM3Tracker
from src.stabilize import stabilize_video_ffmpeg


def run_pipeline(
        video_path: str,
        config_path: str = "config/bodycam.yaml",
        output_path: str = "output/trajectory.json"
) -> None:
    """
        Запуск полного пайплайна обработки видео и сохранения траектории.

        Parameters
        ----------
        video_path : str
            Путь к входному видеофайлу (MP4/AVI).
        config_path : str, optional
            Путь к YAML-конфигу камеры (fx, width, height, ORB-параметры), по умолчанию "config/bodycam.yaml".
        output_path : str, optional
            Путь для сохранения результата в формате JSON, по умолчанию "output/trajectory.json".

        Raises
        ------
        FileNotFoundError
            Если видео не найдено или не может быть открыто.

        Notes
        -----
        Алгоритм:

        0. Стабилизируем видео
        1. Открытие видео через ``cv2.VideoCapture``.
        2. Извлечение FPS и общего количества кадров.
        3. Инициализация:
           - ``MiDaSEstimator`` — для оценки глубины.
           - ``ORBSLAM3Tracker`` — для SLAM и построения траектории.
        4. По кадрам:
           - Оценка средней глубины (``mean_depth``).
           - Трекинг позы с масштабированием по глубине.
           - Сбор траектории и глубин.
        5. После обработки:
           - Расчёт общей дистанции (сумма евклидовых расстояний между точками).
           - Расчёт средней скорости в км/ч.
           - Формирование и сохранение JSON-результата.

        Внимание:
            - Обработка последовательная (1 видео за раз).
            - В прототипе используется CPU для MiDaS. В продакшене — device="cuda".
            - Watchdog для автообнаружения файлов будет добавлен позже.

        Examples
        --------
        Пример входного JSON файла:

        | {
        |     "video": "path/to/video.mp4",
        |     "duration_sec": 3600.0,
        |     "distance_m": 2500.5,
        |     "avg_speed_kmh": 5.2,
        |     "points": 12345,
        |     "avg_depth_m": 3.4,
        |     "trajectory": [[0,0,0], [1.2,0.5,0.1], [2.4,0.8,0.2]]
        | }

    """
    # 0. Стабилизация
    # Настраиваем
    stable_path = "data/temp/stabilized.mp4"
    video_path = stabilize_video_ffmpeg(video_path, stable_path)

    # 1. Открытие стабилизированного видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0
    print(f"Видео: {total_frames} кадров, {duration:.1f} сек, {fps:.1f} FPS")

    # 2. Инициализация моделей (один раз)
    midas = MiDaSEstimator(device="cpu")  # или "cuda"
    slam = ORBSLAM3Tracker(vocab_path="unused", config_path=config_path)

    trajectory = [[0.0, 0.0, 0.0]]  # стартовая точка
    prev_depth = None

    # 3. Обработка кадров
    with tqdm(total=total_frames, desc="Обработка") as pbar:
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Глубина (MiDaS)
            _, mean_depth = midas.predict(frame)
            print(f"Frame {frame_id}: mean_depth={mean_depth:.2f}m")

            # Масштаб
            if prev_depth is not None and prev_depth > 0:
                rel_scale = prev_depth / mean_depth
            else:
                rel_scale = 1.0
            prev_depth = mean_depth
            scale = mean_depth * rel_scale

            # SLAM
            pose = slam.track(frame_rgb, frame_id / fps, scale_factor=scale)
            if pose is not None:
                trajectory.append(pose)

            frame_id += 1
            pbar.update(1)

    # 4. Освобождение ресурсов
    cap.release()
    slam.shutdown()

    # 5. Расчёт дистанции
    if len(trajectory) > 1:
        diffs = np.diff(np.array(trajectory), axis=0)
        segment_distances = np.linalg.norm(diffs, axis=1)
        total_distance = float(segment_distances.sum())
        avg_speed_kmh = total_distance / duration * 3.6 if duration > 0 else 0.0
    else:
        total_distance = 0.0
        avg_speed_kmh = 0.

    # 6. Результат
    result = {
        "video": video_path,
        "duration_sec": round(duration, 2),
        "distance_m": round(total_distance, 2),
        "avg_speed_kmh": round(avg_speed_kmh, 2),
        "points": len(trajectory),
        "trajectory": trajectory
    }

    # 7. Сохранение
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nГОТОВО! Пройдено: {result['distance_m']} м → {output_path}")