import json
import numpy as np
from pathlib import Path


def calculate_scale_factor(real_distance, measured_distance):
    """Рассчет коэффициента масштабирования"""
    if measured_distance == 0:
        return 1.0
    return real_distance / measured_distance


def analyze_trajectory(json_path):
    """Анализ траектории из JSON файла"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        trajectory = data['trajectory_data']['points']

        # Рассчитываем расстояние по точкам
        measured_distance = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i]['x'] - trajectory[i - 1]['x']
            dy = trajectory[i]['y'] - trajectory[i - 1]['y']
            dz = trajectory[i]['z'] - trajectory[i - 1]['z']
            segment_distance = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            measured_distance += segment_distance

        print("📊 АНАЛИЗ ТРАЕКТОРИИ:")
        print(f"Точек траектории: {len(trajectory)}")
        print(f"Измеренное расстояние: {measured_distance:.2f} единиц")

        # Анализ направлений
        x_coords = [p['x'] for p in trajectory]
        y_coords = [p['y'] for p in trajectory]

        print(f"Диапазон X: {min(x_coords):.2f} до {max(x_coords):.2f}")
        print(f"Диапазон Y: {min(y_coords):.2f} до {max(y_coords):.2f}")

        # Анализ поворотов если есть
        if 'turn_analysis' in data:
            turns = data['turn_analysis']['turns']
            print(f"Обнаружено поворотов: {len(turns)}")
            for turn in turns:
                print(f"  - {turn['turn_type']} поворот: {abs(turn['angle_degrees']):.1f}°")

        return measured_distance

    except Exception as e:
        print(f"Ошибка анализа файла: {e}")
        return 0.0


def main():
    print("🎯 КАЛИБРОВКА МАСШТАБА SLAM СИСТЕМЫ")
    print("=" * 50)

    # Параметры калибровки для вашего видео
    REAL_DISTANCE = 82.0  # Реальное расстояние в метрах из вашего описания

    # Анализ существующего файла
    json_file = "../data/output/forest_walk_analysis.json"

    if Path(json_file).exists():
        measured = analyze_trajectory(json_file)
        scale_factor = calculate_scale_factor(REAL_DISTANCE, measured)

        print(f"\n🎯 РЕКОМЕНДУЕМЫЕ НАСТРОЙКИ:")
        print(f"Реальное расстояние: {REAL_DISTANCE} м")
        print(f"Измеренное SLAM: {measured:.2f} единиц")
        print(f"КОЭФФИЦИЕНТ МАСШТАБИРОВАНИЯ: {scale_factor:.3f}")

        print(f"\n💡 ДЛЯ ИСПОЛЬЗОВАНИЯ:")
        print(f'processor = FullFeatureProcessor(input_dir, output_dir, scale_factor={scale_factor:.3f})')

        # Тестовый расчет для разных расстояний
        print(f"\n📏 ТЕСТОВЫЕ РАСЧЕТЫ ДЛЯ МАСШТАБА {scale_factor:.3f}:")
        test_distances = [10, 20, 50, 100]
        print("Реальное → SLAM единицы:")
        for dist in test_distances:
            scaled = dist / scale_factor
            print(f"  {dist:3d} м → {scaled:6.2f} единиц SLAM")

        print(f"\n🚀 СОВЕТЫ:")
        print("1. Используйте этот коэффициент в FullFeatureProcessor")
        print("2. Для точной калибровки снимите видео с известным расстоянием 10м")
        print("3. Перезапустите обработку того же видео с новым масштабом")

    else:
        print(f"❌ Файл {json_file} не найден")
        print("\n📹 Сначала обработайте видео:")
        print("1. Положите видео в data/input/")
        print("2. Запустите систему: python run_slam.py")
        print("3. Дождитесь обработки")
        print("4. Запустите калибровку снова")


if __name__ == "__main__":
    main()