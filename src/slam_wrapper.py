import cv2
import numpy as np
import time


class HighAccuracyVisualOdometry:
    """Визуальный одометр с повышенной точностью"""

    def __init__(self, use_deep_learning=True, scale_factor=1.0):
        self.use_deep_learning = use_deep_learning
        self.scale_factor = scale_factor

        # УВЕЛИЧЕННОЕ количество features для большей точности
        self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.1, nlevels=8, edgeThreshold=15)

        # Более строгий матчинг
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=12,  # Увеличил
                            key_size=20,  # Увеличил
                            multi_probe_level=2)  # Увеличил
        search_params = dict(checks=100)  # Больше проверок
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Калибровка камеры (упрощенная)
        self.camera_matrix = np.array([[800, 0, 320],
                                       [0, 800, 180],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros(4)

        # Улучшенная инициализация
        self.trajectory = [[0.0, 0.0, 0.0]]
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
        self.frame_count = 0
        self.turn_points = []
        self.processing_times = []

        # Для сглаживания траектории
        self.pose_buffer = []
        self.buffer_size = 5

    def process_frame(self, frame):
        start_time = time.time()
        self.frame_count += 1

        # УЛУЧШЕННАЯ предобработка
        processed_frame = self._enhanced_preprocess(frame)
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

        # Детекция с лучшими параметрами
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_frame is not None and des is not None and self.prev_des is not None:
            try:
                # СТРОГИЙ матчинг
                matches = self.flann.knnMatch(self.prev_des, des, k=2)

                # Жесткий фильтр Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.6 * n.distance:  # Более строгий порог
                            good_matches.append(m)

                # МИНИМУМ совпадений увеличен
                if len(good_matches) > 30:  # Было 20
                    src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
                    dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches])

                    # УЛУЧШЕННЫЙ RANSAC
                    M, mask = cv2.estimateAffinePartial2D(
                        src_pts, dst_pts,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=2.0,  # Более строгий
                        confidence=0.995,  # Выше уверенность
                        maxIters=2000  # Больше итераций
                    )

                    if M is not None:
                        inlier_count = np.sum(mask)
                        inlier_ratio = inlier_count / len(good_matches)

                        # Только при хорошем качестве совпадений
                        if inlier_ratio > 0.6 and inlier_count > 20:  # Строже
                            # УЛУЧШЕННОЕ масштабирование
                            dx = M[0, 2] * 0.003 * self.scale_factor  # Более точный коэффициент
                            dy = M[1, 2] * 0.003 * self.scale_factor

                            rotation = np.arctan2(M[1, 0], M[0, 0])

                            # СГЛАЖИВАНИЕ траектории
                            new_pos = self._smooth_trajectory([
                                self.trajectory[-1][0] + dx,
                                self.trajectory[-1][1] + dy,
                                self.trajectory[-1][2] + rotation * 0.05  # Меньше влияние вращения
                            ])

                            self.trajectory.append(new_pos)
                            self._detect_turns_improved()

            except Exception as e:
                print(f"Ошибка обработки кадра {self.frame_count}: {e}")
                # Консервативный подход - минимальное движение
                self.trajectory.append(self.trajectory[-1].copy())

        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_des = des

        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)

        return self.trajectory[-1]

    def _enhanced_preprocess(self, frame):
        """Улучшенная предобработка кадра"""
        # Сохраняем оригинальное разрешение для точности
        if frame.shape[1] > 960:  # Меньше ресайз для точности
            frame = cv2.resize(frame, (960, 540))

        # Улучшенное повышение контраста
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Легкое шумоподавление
        frame = cv2.medianBlur(frame, 3)

        return frame

    def _smooth_trajectory(self, new_pos):
        """Сглаживание траектории скользящим средним"""
        self.pose_buffer.append(new_pos)
        if len(self.pose_buffer) > self.buffer_size:
            self.pose_buffer.pop(0)

        # Среднее по буферу
        smoothed = np.mean(self.pose_buffer, axis=0)
        return smoothed.tolist()

    def _detect_turns_improved(self, window_size=20):  # Увеличил окно
        """Улучшенная детекция поворотов"""
        if len(self.trajectory) < window_size + 1:
            return

        i = len(self.trajectory) - 1

        # Используем больше кадров для стабильности
        start_idx = max(0, i - window_size)
        mid_idx = start_idx + window_size // 2

        # Векторы до и после
        vec_before = [
            self.trajectory[mid_idx][0] - self.trajectory[start_idx][0],
            self.trajectory[mid_idx][1] - self.trajectory[start_idx][1]
        ]

        vec_after = [
            self.trajectory[i][0] - self.trajectory[mid_idx][0],
            self.trajectory[i][1] - self.trajectory[mid_idx][1]
        ]

        # Нормализуем векторы
        norm_before = np.linalg.norm(vec_before)
        norm_after = np.linalg.norm(vec_after)

        if norm_before > 0.1 and norm_after > 0.1:  # Минимальное движение
            vec_before = [v / norm_before for v in vec_before]
            vec_after = [v / norm_after for v in vec_after]

            # Угол через скалярное произведение
            dot_product = vec_before[0] * vec_after[0] + vec_before[1] * vec_after[1]
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            angle_deg = np.degrees(angle_rad)

            # Определяем направление через векторное произведение
            cross_product = vec_before[0] * vec_after[1] - vec_before[1] * vec_after[0]
            turn_type = 'left' if cross_product > 0 else 'right'

            # Более строгий порог для поворотов
            if angle_deg > 20:  # Было 15
                turn_info = {
                    'frame_index': self.frame_count,
                    'trajectory_index': i,
                    'angle_degrees': round(angle_deg, 1),
                    'position': self.trajectory[i].copy(),
                    'turn_type': turn_type
                }

                # Проверка на дубликаты
                if not self.turn_points or abs(i - self.turn_points[-1]['trajectory_index']) > 15:
                    self.turn_points.append(turn_info)
                    print(f"🔄 Обнаружен поворот: {turn_info['turn_type']} {angle_deg:.1f}°")

    # Остальные методы остаются
    def set_scale_factor(self, scale_factor):
        self.scale_factor = scale_factor
        print(f"📏 Установлен масштаб: {scale_factor}")

    def get_trajectory(self):
        return self.trajectory

    def get_turn_points(self):
        return self.turn_points

    def get_statistics(self):
        if not self.processing_times:
            return {}

        total_distance = self._calculate_distance(self.trajectory)

        return {
            'total_frames': self.frame_count,
            'trajectory_points': len(self.trajectory),
            'estimated_distance': total_distance,
            'avg_processing_time': np.mean(self.processing_times),
            'total_processing_time': np.sum(self.processing_times),
            'fps': 1.0 / np.mean(self.processing_times) if np.mean(self.processing_times) > 0 else 0,
            'scale_factor': self.scale_factor,
            'turns_detected': len(self.turn_points)
        }

    def _calculate_distance(self, trajectory):
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

    def reset(self):
        self.__init__(use_deep_learning=self.use_deep_learning, scale_factor=self.scale_factor)