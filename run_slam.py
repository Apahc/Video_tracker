import sys
import os
import argparse
from pathlib import Path

# Добавляем src в путь Python для корректных импортов
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from processor import FullFeatureProcessor


def process_single_video(video_filename, scale_factor=3.35):
    """Обработка конкретного видеофайла"""

    input_dir = "../data/input"
    output_dir = "../data/output"

    # Создаем директории
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Полный путь к файлу
    video_path = Path(input_dir) / video_filename

    if not video_path.exists():
        print(f"❌ Файл не найден: {video_path}")
        print(f"📁 Положите файл в папку: {input_dir}")
        return False

    print(f"🚀 Запуск обработки файла: {video_filename}")
    print(f"📏 Масштаб: {scale_factor}")
    print("-" * 50)

    try:
        # Создаем процессор и обрабатываем файл
        processor = FullFeatureProcessor(input_dir, output_dir, scale_factor=scale_factor)
        result = processor.process_video(video_path)

        if result:
            print(f"✅ Обработка завершена успешно!")
            print(f"📊 Результаты сохранены в: {output_dir}")
            return True
        else:
            print("❌ Ошибка обработки")
            return False

    except Exception as e:
        print(f"💥 Ошибка: {e}")
        return False


def list_input_files():
    """Показать файлы в папке input"""
    input_dir = Path("../data/input")
    if input_dir.exists():
        video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi")) + list(input_dir.glob("*.mov"))
        if video_files:
            print("📁 Доступные видеофайлы:")
            for i, file in enumerate(video_files, 1):
                print(f"  {i}. {file.name}")
        else:
            print("📁 Папка data/input пуста")
    else:
        print("📁 Папка data/input не существует")


def main():
    parser = argparse.ArgumentParser(description='SLAM обработка видео')
    parser.add_argument('filename', nargs='?', help='Имя видеофайла в папке data/input')
    parser.add_argument('--scale', type=float, default=3.35, help='Коэффициент масштабирования')
    parser.add_argument('--list', action='store_true', help='Показать доступные файлы')

    args = parser.parse_args()

    print("🎯 SLAM СИСТЕМА - ОБРАБОТКА ВИДЕО")
    print("=" * 50)

    if args.list:
        list_input_files()
        return

    if args.filename:
        # Обработка указанного файла
        process_single_video(args.filename, args.scale)
    else:
        # Интерактивный режим
        list_input_files()
        print("\n💡 Использование:")
        print("  python run_slam.py filename.mp4")
        print("  python run_slam.py filename.mp4 --scale 2.5")
        print("  python run_slam.py --list")

        filename = input("\n📹 Введите имя файла для обработки: ").strip()
        if filename:
            scale_input = input("📏 Введите масштаб (Enter для 3.35): ").strip()
            scale_factor = float(scale_input) if scale_input else 3.35
            process_single_video(filename, scale_factor)
        else:
            print("❌ Не указан файл для обработки")


if __name__ == "__main__":
    main()