# src/pipeline.py
import cv2
import json
import numpy as np
from tqdm import tqdm
from .midas import MiDaSEstimator
from .slam import ORBSLAM3Tracker

def run_pipeline(
    video_path: str,
    config_path: str = "config/bodycam.yaml",
    output_path: str = "output/trajectory.json"
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"Видео: {total_frames} кадров, {duration:.1f} сек, {fps:.1f} FPS")

    # ИНИЦИАЛИЗАЦИЯ ОДИН РАЗ!
    midas = MiDaSEstimator(device="cpu")  # или "cuda"
    slam = ORBSLAM3Tracker(vocab_path="unused", config_path=config_path)

    trajectory = []
    depths = []

    with tqdm(total=total_frames, desc="Обработка") as pbar:
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Глубина
            _, mean_depth = midas.predict(frame)
            depths.append(mean_depth)

            # SLAM
            pose = slam.track(frame_rgb, frame_id / fps, scale_factor=mean_depth)
            if pose is not None:
                trajectory.append(pose)

            frame_id += 1
            pbar.update(1)

    cap.release()
    slam.shutdown()

    # === РЕЗУЛЬТАТ ===
    total_distance = np.linalg.norm(np.diff(np.array(trajectory), axis=0), axis=1).sum()
    result = {
        "video": video_path,
        "duration_sec": duration,
        "distance_m": round(total_distance, 2),
        "avg_speed_kmh": round(total_distance / duration * 3.6, 2),
        "points": len(trajectory),
        "avg_depth_m": round(np.mean(depths), 2),
        "trajectory": trajectory
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nГОТОВО! Пройдено: {result['distance_m']} м → {output_path}")