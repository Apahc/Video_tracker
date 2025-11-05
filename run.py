"""
Точка входа для запуска оффлайн-обработки видео через командную строку.

Модуль предоставляет CLI-интерфейс для вызова основного пайплайна.
Используется для тестирования, отладки
и ручного запуска обработки одного видеофайла.

Путь до данного файла: run.py
"""
import argparse
from src.pipeline import run_pipeline


def main() -> None:
    """
    Основная функция: парсинг аргументов и запуск пайплайна.

    Notes
    -----
    Аргументы командной строки:
        --input     Путь к входному видео (по умолчанию: data/input/my_video_2.mp4).
        --output    Путь к выходному JSON (по умолчанию: output/trajectory.json).
        --config    Путь к конфигу камеры и ORB (по умолчанию: config/bodycam.yaml).

    Examples
    --------
        python run.py --input data/input/bodycam_10h.mp4 --output output/cam01.json
    """
    parser = argparse.ArgumentParser(
        description="Запуск оффлайн-анализа траектории движения по видео с bodycam"
    )
    parser.add_argument(
        "--input",
        default="data/input/my_video_2.mp4",
        help="Путь к входному видеофайлу (MP4/AVI)"
    )
    parser.add_argument(
        "--output",
        default="output/trajectory.json",
        help="Путь для сохранения результата в JSON"
    )
    parser.add_argument(
        "--config",
        default="config/bodycam.yaml",
        help="Путь к YAML-конфигу (камера + ORB-параметры)"
    )
    args = parser.parse_args()

    # УДАЛЕНИЕ vocab_path: в упрощённой версии SLAM словарь не используется
    # (в полной ORB-SLAM3 — нужен для BoW и loop closure)
    print(f"Запуск обработки: {args.input}")
    print(f"Конфиг: {args.config}")
    print(f"Вывод: {args.output}")

    run_pipeline(
        video_path=args.input,
        config_path=args.config,
        output_path=args.output
    )

    print("Обработка завершена.")


if __name__ == "__main__":
    main()
