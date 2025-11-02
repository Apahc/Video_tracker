# smart_trajectory_flex.py
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter
import argparse
from pathlib import Path
from typing import List, Tuple


# === JSON-safe ===
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


# === Загрузка ===
def load_trajectory(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    points = np.array(data["trajectory"], dtype=np.float64)
    return points


# === Удаление выбросов ===
def remove_outliers_iqr(points, factor):
    if len(points) < 5: return points
    diffs = np.diff(points, axis=0)
    speeds = np.linalg.norm(diffs, axis=1)
    Q1, Q3 = np.percentile(speeds, [25, 75])
    IQR = Q3 - Q1
    mask = np.concatenate([[True], (speeds >= Q1 - factor * IQR) & (speeds <= Q3 + factor * IQR)])
    filtered = points[mask]
    print(f"Удалено выбросов: {len(points) - len(filtered)} → {len(filtered)} точек")
    return filtered


# === SavGol ===
def smooth_savgol(points, window_ratio):
    n = len(points)
    window = max(5, min(n - 1 if n % 2 == 0 else n, int(n * window_ratio)))
    window = window if window % 2 == 1 else window - 1
    if window < 5: return points
    return savgol_filter(points.T, window_length=window, polyorder=3).T


# === Сплайн ===
def refine_spline(points, duration, max_kmh, num_points):
    if len(points) < 4: return points
    x, y, z = points.T
    t = np.arange(len(x))
    s = len(points) * 1e-3

    try:
        tck, u = splprep([x, y, z], u=t, s=s, k=3)
        u_new = np.linspace(0, u[-1], num_points)
        refined = np.column_stack(splev(u_new, tck))

        dt = duration / (num_points - 1)
        speeds = np.linalg.norm(np.diff(refined, axis=0), axis=1) / dt * 3.6
        max_speed = np.max(speeds)
        if max_speed > max_kmh:
            print(f"Скорость ограничена: {max_speed:.1f} → {max_kmh} км/ч")
        return refined, max_speed
    except Exception as e:
        print(f"Сплайн не удался: {e}")
        return points, 0.0


# === Парсинг эталонного пути ===
def parse_reference_path(path_str: str) -> Tuple[float, List[Tuple[float, float]]]:
    segments = []
    total_distance = 0.0
    current_angle = 0.0
    x, y = 0.0, 0.0
    points = [(x, y)]

    for part in path_str.split(','):
        part = part.strip()
        if not part: continue
        if part.endswith(('L', 'R')):
            angle_change = float(part[:-1])
            direction = -1 if part.endswith('L') else 1
            current_angle += direction * np.radians(angle_change)
        else:
            dist = float(part)
            dx = dist * np.cos(current_angle)
            dy = dist * np.sin(current_angle)
            x += dx
            y += dy
            points.append((x, y))
            total_distance += dist

    return total_distance, points


# === Валидация ===
def validate_against_path(real_dist, expected_dist):
    error = abs(real_dist - expected_dist) / expected_dist * 100
    if error < 20:
        return f"ОТЛИЧНО! Ошибка {error:.1f}%"
    elif error < 50:
        return f"Хорошо, ошибка {error:.1f}%"
    else:
        return f"Проверь масштаб! Ошибка {error:.1f}%"


# === Графики ===
def plot_result(raw, refined, info, duration, max_speed, ref_dist, ref_points):
    fig = plt.figure(figsize=(20, 10))

    # 1. XY + эталон
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(raw[:, 0], raw[:, 1], 'r-', alpha=0.5, label='Raw', linewidth=1)
    ax1.plot(refined[:, 0], refined[:, 1], 'g-', linewidth=3, label='Smoothed')
    if ref_points:
        ref_x, ref_y = zip(*ref_points)
        ax1.plot(ref_x, ref_y, 'b--', linewidth=2, label='Эталон')
    ax1.set_xlabel('X (м)');
    ax1.set_ylabel('Y (м)')
    ax1.axis('equal');
    ax1.grid(True, alpha=0.3)
    ax1.legend();
    ax1.set_title("Траектория (XY)")

    # 2. 3D
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.plot(raw[:, 0], raw[:, 1], raw[:, 2], 'r-', alpha=0.5)
    ax2.plot(refined[:, 0], refined[:, 1], refined[:, 2], 'g-', linewidth=2)
    ax2.set_title("3D")

    # 3. Скорость
    ax3 = fig.add_subplot(2, 3, 3)
    dt = duration / (len(refined) - 1)
    speed = np.linalg.norm(np.diff(refined, axis=0), axis=1) / dt * 3.6
    ax3.plot(speed, 'g-', linewidth=2)
    ax3.axhline(info["avg_speed_kmh"], color='blue', linestyle='--', label=f'Средняя: {info["avg_speed_kmh"]} км/ч')
    ax3.axhline(max_speed, color='red', linestyle=':', label=f'Макс: {max_speed:.1f} км/ч')
    ax3.set_ylim(0, max_speed * 1.5);
    ax3.legend();
    ax3.grid(True)
    ax3.set_title("Скорость")

    # 4. Инфо
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axis('off')
    text = f"""
    АНАЛИЗ
    ━━━━━━━━━━━━━━━━━
    Дистанция:     {info["distance_m"]} м
    Эталон:        {ref_dist:.1f} м
    Скорость:      {info["avg_speed_kmh"]} км/ч
    Макс:          {max_speed:.1f} км/ч
    Поворотов:     {info["turn_count"]}
    Шум:           {info["noise_m_per_s"]} м/с
    Валидация:     {validate_against_path(info["distance_m"], ref_dist)}
    """
    ax4.text(0.1, 0.9, text, fontsize=11, verticalalignment='top', fontfamily='monospace')

    # 5. Гистограмма
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(speed, bins=20, alpha=0.7, color='lightgreen')
    ax5.axvline(info["avg_speed_kmh"], color='blue', linestyle='--')
    ax5.set_xlabel('Скорость (км/ч)');
    ax5.set_title("Распределение")

    # 6. Легенда поворотов
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    ax6.text(0.1, 0.9, "Условные обозначения:\n90L = поворот налево 90°\n90R = поворот направо 90°\n180 = разворот",
             fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.suptitle("SMART TRAJECTORY v5.0 — ГИБКИЙ АНАЛИЗ", fontsize=16)
    plt.tight_layout()
    plt.show()


# === Главная ===
def main():
    parser = argparse.ArgumentParser(description="Гибкий анализ траектории")
    parser.add_argument('--input', type=str,default='output/trajectory.json',  help='Путь к trajectory.json')
    parser.add_argument('--duration', type=float,default='30',  help='Длительность видео (сек)')
    parser.add_argument('--path', type=str, default="5,180L,5,90L,3,180R,2,90R,4,90L,1,90R,5",
                        help='Эталон: "5,180,5,90L,3,180,2,90R,4,90L,1,90R,5"')
    parser.add_argument('--max_speed', type=float,default='10', help='Макс. скорость (км/ч)')
    parser.add_argument('--outlier_factor', type=float, default=2.5)
    parser.add_argument('--smooth_window', type=float, default=0.2)
    parser.add_argument('--num_points', type=int, default=300)
    args = parser.parse_args()

    print(f"Запуск анализа для видео {args.duration} сек...")
    raw = load_trajectory(args.input)
    print(f"Вход: {len(raw)} точек")

    # Эталон
    ref_dist, ref_points = parse_reference_path(args.path)
    print(f"Эталонная длина: {ref_dist:.1f} м")

    # Обработка
    clean = remove_outliers_iqr(raw, factor=args.outlier_factor)
    smooth = smooth_savgol(clean, window_ratio=args.smooth_window)
    refined, max_speed = refine_spline(smooth, args.duration, args.max_speed, args.num_points)

    # Метрики
    final_dist = np.sum(np.linalg.norm(np.diff(refined, axis=0), axis=1))
    avg_speed = final_dist / args.duration * 3.6

    # Повороты
    vecs = np.diff(smooth, axis=0)
    turn_count = 0
    if len(vecs) >= 2:
        angles = np.diff(np.arctan2(vecs[:, 1], vecs[:, 0]))
        angles = np.abs(np.mod(angles + np.pi, 2 * np.pi) - np.pi)
        turn_count = int(np.sum(angles > np.pi / 6))

    info = {
        "distance_m": round(final_dist, 1),
        "avg_speed_kmh": round(avg_speed, 1),
        "noise_m_per_s": round(
            np.std(np.linalg.norm(np.diff(clean, axis=0), axis=1) / (args.duration / (len(clean) - 1))), 2),
        "turn_count": turn_count
    }

    # Сохранение
    result = {
        "duration_sec": args.duration,
        "reference_path_m": ref_dist,
        "distance_m": info["distance_m"],
        "avg_speed_kmh": info["avg_speed_kmh"],
        "max_speed_kmh": max_speed,
        "validation": validate_against_path(info["distance_m"], ref_dist),
        "trajectory": refined.tolist()
    }
    out_path = Path('FLEX_result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"Сохранено в {out_path}")

    # Графики
    plot_result(raw, refined, info, args.duration, max_speed, ref_dist, ref_points)


if __name__ == "__main__":
    main()