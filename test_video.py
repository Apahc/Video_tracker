# test_video.py  # Тест чтения и сжатия видео до 15fps
import cv2
import os
from tqdm import tqdm


def prepare_video(input_path, output_path, target_fps=15, max_duration_sec=300):  # max 5 мин для теста
    """
    Читает видео, сжимает до target_fps (для оптимизации), обрезает до max_duration_sec.
    Возвращает cap для дальнейшей обработки или сохраняет новое видео.
    """
    if not os.path.exists(input_path):
        print(
            f"Ошибка: Видео {input_path} не найдено. Скачай с Pexels: https://www.pexels.com/search/videos/body%20cam/")
        return None

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видео. Проверь формат (MP4/AVI).")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int( fps / target_fps)  # Каждый N-й кадр для 15fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    print(f"Оригинал: {duration_sec:.1f} сек, {fps} FPS, {total_frames} кадров.")

    # Обрезаем до max_duration_sec
    max_frames = int(max_duration_sec * target_fps)
    frames = []

    frame_id = 0
    pbar = tqdm(total=min(total_frames, max_frames))
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

    if frames:
        # Сохраняем новое видео (опционально)
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"Сжатое видео сохранено: {output_path} ({len(frames)} кадров, {len(frames) / target_fps:.1f} сек)")
        return output_path
    return None


# Тест
if __name__ == "__main__":
    input_video = "data/input/my_video_2.mp4"  # Положи скачанное видео сюда
    output_video = "data/output/my_video_2_15fps.mp4"
    prepare_video(input_video, output_video)