import cv2
import numpy as np
import json
import time
import logging
from pathlib import Path

# ИСПРАВЛЕННЫЙ ИМПОРТ
from slam_wrapper import HighAccuracyVisualOdometry

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_video_info(video_path):
    """Получение информации о видео"""
    cap = cv2.VideoCapture(video_path)
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)) if cap.get(
            cv2.CAP_PROP_FPS) > 0 else 0
    }
    cap.release()
    return info


class FullFeatureProcessor:
    """Полнофункциональный процессор видео с повышенной точностью"""

    def __init__(self, input_dir, output_dir, scale_factor=3.35):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ИСПРАВЛЕННАЯ ИНИЦИАЛИЗАЦИЯ
        self.vo = HighAccuracyVisualOdometry(use_deep_learning=True, scale_factor=scale_factor)

        logger.info(f"Инициализирован FullFeatureProcessor с scale_factor={scale_factor}")

    def _calculate_distance(self, trajectory):
        """Вычисление пройденной дистанции"""
        if len(trajectory) < 2:
            return 0.0

        distance = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i - 1][0]
            dy = trajectory[i][1] - trajectory[i - 1][1]
            dz = trajectory[i][2] - trajectory[i - 1][2]
            segment_distance = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            distance += segment_distance

        return distance

    def set_scale_factor(self, scale_factor):
        """Изменение масштаба во время работы"""
        self.vo.set_scale_factor(scale_factor)
        logger.info(f"Установлен scale_factor={scale_factor}")

    def process_video(self, video_path):
        """Обработка конкретного видеофайла"""
        start_time = time.time()
        video_path = Path(video_path)

        if not video_path.exists():
            logger.error(f"Файл не найден: {video_path}")
            return None

        logger.info(f"🚀 Начало обработки: {video_path.name}")

        # Получаем информацию о видео
        video_info = get_video_info(str(video_path))
        logger.info(
            f"📹 Информация о видео: {video_info['width']}x{video_info['height']}, {video_info['fps']:.1f} FPS, {video_info['duration']:.1f} сек")

        # Обработка видео
        cap = cv2.VideoCapture(str(video_path))
        frame_skip = 3  # Обрабатываем каждый 3-й кадр
        frame_count = 0

        print(f"⏳ Обработка видео...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                self.vo.process_frame(frame)

            frame_count += 1

            # Прогресс для длинных видео
            if frame_count % 100 == 0:
                progress = (frame_count / video_info['frame_count']) * 100
                print(f"📊 Прогресс: {frame_count}/{video_info['frame_count']} кадров ({progress:.1f}%)")

        cap.release()

        # Получаем результаты
        trajectory = self.vo.get_trajectory()
        turn_points = self.vo.get_turn_points()
        stats = self.vo.get_statistics()

        # Формируем результат
        result = {
            "method": "advanced_vo_scaled",
            "trajectory": trajectory,
            "turn_points": turn_points,
            "frame_count": frame_count,
            "trajectory_points": len(trajectory),
            "processing_stats": stats,
            "total_processing_time": time.time() - start_time,
            "video_info": video_info
        }

        # Сохраняем результаты
        self._save_detailed_results(video_path, result)

        logger.info(f"✅ Обработка завершена: {video_path.name}")
        logger.info(f"📊 Результаты: {result['trajectory_points']} точек траектории")
        logger.info(f"📏 Дистанция: {stats['estimated_distance']:.2f} единиц (масштаб: {stats['scale_factor']})")
        logger.info(f"🔄 Обнаружено поворотов: {len(turn_points)}")

        return result

    def _save_detailed_results(self, video_path, result):
        """Сохранение детализированных результатов с информацией о поворотах"""

        # Подготовка данных о поворотах
        turn_data = []
        for turn in result["turn_points"]:
            turn_data.append({
                "frame_index": turn["frame_index"],
                "trajectory_index": turn["trajectory_index"],
                "angle_degrees": turn["angle_degrees"],
                "position": {
                    "x": round(turn["position"][0], 4),
                    "y": round(turn["position"][1], 4),
                    "z": round(turn["position"][2], 4)
                },
                "turn_type": turn["turn_type"]
            })

        output_data = {
            "analysis_info": {
                "camera_id": video_path.stem,
                "video_file": str(video_path),
                "processing_method": result["method"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": "2.0"
            },
            "video_statistics": {
                "total_frames": result["frame_count"],
                "trajectory_points": result["trajectory_points"],
                "estimated_distance": round(result["processing_stats"]["estimated_distance"], 3),
                "total_processing_time": round(result["total_processing_time"], 2),
                "processing_fps": round(result["processing_stats"].get('fps', 0), 1),
                "scale_factor": result["processing_stats"]["scale_factor"],
                "turns_detected": len(result["turn_points"])
            },
            "trajectory_data": {
                "points": [{"x": round(p[0], 4), "y": round(p[1], 4), "z": round(p[2], 4)}
                           for p in result["trajectory"]]
            },
            "turn_analysis": {
                "turns": turn_data,
                "total_turns": len(turn_data)
            },
            "processing_details": result["processing_stats"]
        }

        # Сохраняем JSON
        output_path = self.output_dir / f"{video_path.stem}_analysis.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # Создаем улучшенную визуализацию
        self._create_enhanced_visualization(result["trajectory"], result["turn_points"], video_path.stem)

        logger.info(f"💾 Результаты сохранены: {output_path}")
        print(f"💾 Результаты сохранены: {output_path}")

    def _create_enhanced_visualization(self, trajectory, turn_points, video_name):
        """Создание визуализации траектории с легендой вне графика"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Конвертируем в метры
            x = [p[0] for p in trajectory]
            y = [p[1] for p in trajectory]

            # РАССЧИТЫВАЕМ оптимальный размер фигуры
            x_range = max(x) - min(x)
            y_range = max(y) - min(y)

            # УВЕЛИЧИВАЕМ ширину для легенды справа
            if max(x_range, y_range) > 100:
                fig_size = (22, 16)  # Шире для легенды
            elif max(x_range, y_range) > 50:
                fig_size = (18, 12)
            elif max(x_range, y_range) > 20:
                fig_size = (16, 10)
            else:
                fig_size = (14, 8)

            # СОЗДАЕМ фигуру с дополнительным местом для легенды
            fig = plt.figure(figsize=fig_size)

            # СОЗДАЕМ сетку: основной график занимает 75%, легенда - 25%
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])

            # Основной график траектории (занимает левую часть)
            ax1 = plt.subplot(gs[0, 0])

            # ОСНОВНОЙ ПРОЦЕСС: строим траекторию
            line, = ax1.plot(x, y, 'b-', alpha=0.7, linewidth=2)
            start_point, = ax1.plot(x[0], y[0], 'go', markersize=8)
            end_point, = ax1.plot(x[-1], y[-1], 'ro', markersize=8)

            scatter_handles = []
            if turn_points:
                turn_x = [turn['position'][0] for turn in turn_points]
                turn_y = [turn['position'][1] for turn in turn_points]

                colors = ['orange' if turn['turn_type'] == 'left' else 'purple' for turn in turn_points]
                scatter = ax1.scatter(turn_x, turn_y, c=colors, s=50, alpha=0.9)
                scatter_handles.append(scatter)

                # ПРОСТЫЕ подписи поворотов (только номера)
                for i, turn in enumerate(turn_points):
                    ax1.annotate(f"{i + 1}",
                                 (turn_x[i], turn_y[i]),
                                 xytext=(5, 5), textcoords='offset points',
                                 fontsize=8, fontweight='bold',
                                 bbox=dict(boxstyle="circle,pad=0.2", fc='white', alpha=0.8))

            # НАСТРОЙКА ОСЕЙ основного графика
            ax1.set_aspect('equal', adjustable='datalim')

            # Добавляем запас вокруг данных
            x_margin = x_range * 0.1
            y_margin = y_range * 0.1

            ax1.set_xlim(min(x) - x_margin, max(x) + x_margin)
            ax1.set_ylim(min(y) - y_margin, max(y) + y_margin)

            # Автоматическая сетка
            grid_step = self._calculate_grid_step(max(x_range, y_range))
            ax1.set_xticks(np.arange(np.floor(min(x) - x_margin),
                                     np.ceil(max(x) + x_margin) + grid_step, grid_step))
            ax1.set_yticks(np.arange(np.floor(min(y) - y_margin),
                                     np.ceil(max(y) + y_margin) + grid_step, grid_step))

            ax1.grid(True, alpha=0.3)
            ax1.set_xlabel('Расстояние по X (метры)', fontsize=11)
            ax1.set_ylabel('Расстояние по Y (метры)', fontsize=11)
            ax1.set_title(f'Траектория движения: {video_name}', fontsize=13, fontweight='bold', pad=15)

            # ЛЕГЕНДА ВНЕ ГРАФИКА (справа)
            ax_legend = plt.subplot(gs[0, 1])
            ax_legend.axis('off')  # Скрываем оси

            # Создаем элементы для легенды
            legend_elements = [
                plt.Line2D([0], [0], color='blue', linewidth=3, label='Траектория движения'),
                plt.Line2D([0], [0], marker='o', color='green', markersize=8,
                           label=f'Начало ({x[0]:.1f}, {y[0]:.1f})'),
                plt.Line2D([0], [0], marker='o', color='red', markersize=8,
                           label=f'Конец ({x[-1]:.1f}, {y[-1]:.1f})')
            ]

            if turn_points:
                legend_elements.extend([
                    plt.Line2D([0], [0], marker='o', color='orange', markersize=8,
                               label=f'Левые повороты ({sum(1 for t in turn_points if t["turn_type"] == "left")} шт.)'),
                    plt.Line2D([0], [0], marker='o', color='purple', markersize=8,
                               label=f'Правые повороты ({sum(1 for t in turn_points if t["turn_type"] == "right")} шт.)')
                ])

            # Добавляем общую информацию
            total_distance = self._calculate_distance(trajectory)
            info_text = f"Общее расстояние: {total_distance:.1f} м\n"
            info_text += f"Всего поворотов: {len(turn_points)}\n"
            info_text += f"Точек траектории: {len(trajectory)}"

            ax_legend.text(0.1, 0.8, info_text, transform=ax_legend.transAxes,
                           fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.5", fc='lightblue', alpha=0.7))

            # РАЗМЕЩАЕМ ЛЕГЕНДУ
            ax_legend.legend(handles=legend_elements,
                             loc='center left',
                             fontsize=10,
                             framealpha=0.9,
                             fancybox=True,
                             shadow=True)

            # График поворотов (второй ряд)
            ax2 = plt.subplot(gs[1, :])  # Занимает всю ширину снизу

            if turn_points:
                turn_numbers = list(range(1, len(turn_points) + 1))
                turn_angles = [turn['angle_degrees'] for turn in turn_points]

                colors = ['orange' if turn['turn_type'] == 'left' else 'purple' for turn in turn_points]
                bars = ax2.bar(turn_numbers, turn_angles, color=colors, alpha=0.7, width=0.7)

                # Подписи значений
                for bar, angle in zip(bars, turn_angles):
                    height = bar.get_height()
                    va = 'bottom' if height >= 0 else 'top'
                    offset = 1 if height >= 0 else -1
                    ax2.text(bar.get_x() + bar.get_width() / 2, height + offset,
                             f'{angle:.0f}°', ha='center', va=va, fontsize=9, fontweight='bold')

                ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax2.set_xlabel('Номер поворота', fontsize=11)
                ax2.set_ylabel('Угол поворота (°)', fontsize=11)
                ax2.set_title('Углы обнаруженных поворотов', fontsize=13, fontweight='bold', pad=15)
                ax2.grid(True, alpha=0.3, axis='y')
                ax2.set_xticks(turn_numbers)

            else:
                ax2.text(0.5, 0.5, 'Повороты не обнаружены',
                         ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_xticks([])
                ax2.set_yticks([])

            # ГАРАНТИРУЕМ, что все помещается
            plt.tight_layout(pad=3.0)

            # Сохраняем
            plot_path = self.output_dir / f"{video_name}_trajectory_enhanced.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight',
                        facecolor='white', edgecolor='none',
                        pad_inches=0.3)

            plt.close()

            print(f"📈 График сохранен: {plot_path}")
            print(f"📍 Легенда вынесена за пределы графика")

        except Exception as e:
            print(f"⚠️ Ошибка создания графика: {e}")

    def _calculate_grid_step(self, range_size):
        """Рассчитывает оптимальный шаг сетки"""
        if range_size > 200:
            return 50
        elif range_size > 100:
            return 20
        elif range_size > 50:
            return 10
        elif range_size > 20:
            return 5
        elif range_size > 10:
            return 2
        else:
            return 1

    def _create_text_report(self, trajectory, turn_points, video_name):
        """Создание текстового отчета о траектории"""
        report_path = self.output_dir / f"{video_name}_report.txt"

        total_distance = self._calculate_distance(trajectory)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("📊 ОТЧЕТ О ТРАЕКТОРИИ ДВИЖЕНИЯ\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"📹 Видеофайл: {video_name}\n")
            f.write(f"📏 Общее пройденное расстояние: {total_distance:.1f} м\n")
            f.write(f"📍 Начальная точка: ({trajectory[0][0]:.1f}, {trajectory[0][1]:.1f}) м\n")
            f.write(f"🎯 Конечная точка: ({trajectory[-1][0]:.1f}, {trajectory[-1][1]:.1f}) м\n")
            f.write(f"🔄 Обнаружено поворотов: {len(turn_points)}\n\n")

            f.write("🧭 ДЕТАЛИЗИРОВАННАЯ ТРАЕКТОРИЯ:\n")
            f.write("-" * 60 + "\n")

            # Анализ общего направления
            total_dx = trajectory[-1][0] - trajectory[0][0]
            total_dy = trajectory[-1][1] - trajectory[0][1]

            if abs(total_dx) > abs(total_dy):
                main_direction = "Запад" if total_dx < 0 else "Восток"
            else:
                main_direction = "Юг" if total_dy < 0 else "Север"

            f.write(f"Основное направление: {main_direction}\n")
            f.write(f"Смещение: {abs(total_dx):.1f} м по X, {abs(total_dy):.1f} м по Y\n\n")

            # Анализ поворотов
            if turn_points:
                f.write("🔄 ОБНАРУЖЕННЫЕ ПОВОРОТЫ:\n")
                f.write("-" * 60 + "\n")

                for i, turn in enumerate(turn_points, 1):
                    # Вычисляем расстояние от начала до этого поворота
                    dist_to_turn = self._calculate_distance(trajectory[:turn['trajectory_index'] + 1])

                    f.write(f"Поворот {i}:\n")
                    f.write(f"  • Тип: {'↰ Левый' if turn['turn_type'] == 'left' else '↱ Правый'}\n")
                    f.write(f"  • Угол: {abs(turn['angle_degrees']):.1f}°\n")
                    f.write(f"  • Координаты: ({turn['position'][0]:.1f}, {turn['position'][1]:.1f}) м\n")
                    f.write(f"  • Пройдено до поворота: {dist_to_turn:.1f} м\n")

                    # Определяем направление после поворота
                    if i < len(turn_points):
                        next_turn = turn_points[i]
                        dx = next_turn['position'][0] - turn['position'][0]
                        dy = next_turn['position'][1] - turn['position'][1]
                    else:
                        dx = trajectory[-1][0] - turn['position'][0]
                        dy = trajectory[-1][1] - turn['position'][1]

                    # Определяем направление
                    if abs(dx) > abs(dy):
                        direction = "Запад" if dx < 0 else "Восток"
                    else:
                        direction = "Юг" if dy < 0 else "Север"

                    f.write(f"  • Направление после: {direction}\n")
                    f.write("\n")

            # Статистика по квадрантам
            f.write("📈 СТАТИСТИКА ПО КВАДРАНТАМ:\n")
            f.write("-" * 60 + "\n")

            quadrants = {"I": 0, "II": 0, "III": 0, "IV": 0}  # счетчики точек

            for point in trajectory:
                x, y = point[0], point[1]
                if x >= 0 and y >= 0:
                    quadrants["I"] += 1
                elif x < 0 and y >= 0:
                    quadrants["II"] += 1
                elif x < 0 and y < 0:
                    quadrants["III"] += 1
                else:
                    quadrants["IV"] += 1

            total_points = len(trajectory)
            for quad, count in quadrants.items():
                percentage = (count / total_points) * 100
                f.write(f"Квадрант {quad}: {count} точек ({percentage:.1f}%)\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("🎯 ВЫВОДЫ:\n")
            f.write("=" * 60 + "\n")

            # Основные выводы
            f.write(f"• Маршрут составляет {total_distance:.1f} метров\n")
            f.write(f"• Начинается в точке ({trajectory[0][0]:.1f}, {trajectory[0][1]:.1f})\n")
            f.write(f"• Заканчивается в точке ({trajectory[-1][0]:.1f}, {trajectory[-1][1]:.1f})\n")

            if turn_points:
                left_turns = sum(1 for t in turn_points if t['turn_type'] == 'left')
                right_turns = len(turn_points) - left_turns
                f.write(f"• Совершено {left_turns} левых и {right_turns} правых поворотов\n")

                avg_turn_angle = sum(abs(t['angle_degrees']) for t in turn_points) / len(turn_points)
                f.write(f"• Средний угол поворота: {avg_turn_angle:.1f}°\n")

            f.write(f"• Основное направление движения: {main_direction}\n")

            # Определяем самый активный квадрант
            main_quadrant = max(quadrants.items(), key=lambda x: x[1])[0]
            quadrant_names = {"I": "северо-восток", "II": "северо-запад",
                              "III": "юго-запад", "IV": "юго-восток"}
            f.write(f"• Основная зона движения: {quadrant_names[main_quadrant]}\n")

        print(f"📄 Текстовый отчет сохранен: {report_path}")
        logger.info(f"Текстовый отчет сохранен: {report_path}")