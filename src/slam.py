"""
Упрощённая реализация визуального SLAM-трекера на основе ORB-фич и OpenCV.

Модуль реализует монокулярный SLAM для оценки позы камеры (position + orientation)
по последовательности кадров. Используется ORB-детектор и essential matrix + RANSAC
для восстановления движения между кадрами. Глобальная траектория строится путём
накопления относительных трансформаций с масштабированием на основе глубины (MiDaS).

Важно: Это упрощённая версия ORB-SLAM3 без:
- Полного графа оптимизации (Bundle Adjustment),
- Loop closure,
- Ключевых кадров,
- Словаря BoW (Bag-of-Words).

Путь до данного файла: src/slam.py
"""
import cv2
import numpy as np
from pathlib import Path
import yaml

from typing import Tuple, List


class ORBSLAM3Tracker:
    """
        Упрощённый SLAM-трекер на основе ORB-фич и OpenCV.

        Использует последовательный подход: каждый кадр сравнивается с предыдущим,
        восстанавливается относительная поза через Essential Matrix, затем
        масштабируется и добавляется к глобальной траектории.

        Attributes
        ----------
        focal : float
            Фокусное расстояние по X (в пикселях).
        cx, cy : float
            Координаты главного центра (principal point).
        width, height : int
            Разрешение кадра (для валидации).
        orb : cv2.ORB
            ORB-детектор с параметрами из конфига.
        matcher : cv2.BFMatcher
            Brute-force matcher с Hamming-нормой и cross-check.
        last_kp, last_desc : list, np.ndarray
            Ключевые точки и дескрипторы предыдущего кадра.
        R_prev, t_prev : np.ndarray
            Накопленная глобальная ротация и трансляция.
        trajectory : list
            Список поз [[x, y, z], ...] в метрах.
        """
    def __init__(
            self,
            vocab_path: str,
            config_path: str
    ) -> None:
        """
        Инициализация SLAM-трекера.

        Parameters
        ----------
        vocab_path : str
            Путь к словарю ORB (не используется в упрощённой версии).
        config_path : str
            Путь к YAML-конфигу с параметрами камеры и ORB.

        Notes
        -----
        Ожидаемый формат конфигурационного файла (bodycam.yaml):

        camera:

        |  fx: 600.0
        |  width: 1280
        |  height: 720
        |  cx: 640.0   # опционально
        |  cy: 360.0   # опционально
        orb:

        |  n_features: 2000
        |  scale_factor: 1.2
        |  n_levels: 8
        |  ini_th_fast: 20
        """
        self.vocab_path = Path(vocab_path)      # Не используется в прототипе
        self.config_path = Path(config_path)

        # Загрузка конфигурации
        with open(self.config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        cam = cfg['camera']
        orb_cfg = cfg['orb']

        # Параметры камеры
        self.focal = float(cam['fx'])
        self.width = int(cam['width'])
        self.height = int(cam['height'])
        self.cx = float(cam.get('cx', self.width / 2))
        self.cy = float(cam.get('cy', self.height / 2))

        # Параметры ORB
        self.n_features = int(orb_cfg['n_features'])
        self.scale_factor = float(orb_cfg['scale_factor'])
        self.n_levels = int(orb_cfg['n_levels'])
        self.ini_th_fast = int(orb_cfg['ini_th_fast'])

        # OpenCV ORB детектор
        self.orb = cv2.ORB_create(
            nfeatures=self.n_features,
            scaleFactor=self.scale_factor,
            nlevels=self.n_levels,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=self.ini_th_fast
        )
        print("[SLAM] Используется OpenCV ORB")

        # Матчинг и состояние
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.last_kp = None
        self.last_desc = None
        self.trajectory = [[0.0, 0.0, 0.0]]
        self.R_prev = np.eye(3, dtype=np.float32)
        self.t_prev = np.zeros((3, 1), dtype=np.float32)

        print(f"[SLAM] Инициализирован: {self.width}x{self.height}, f={self.focal}")

    def _extract(
            self,
            gray: np.ndarray
    ) -> Tuple[list, np.ndarray]:
        """
        Извлечение ORB-фич из grayscale-кадра.

        Parameters
        ----------
        gray : np.ndarray
            Grayscale изображение, форма [H, W], тип uint8.

        Returns
        -------
        kp : list
            Список cv2.KeyPoint.
        desc : np.ndarray
            Дескрипторы, форма [N, 32], тип uint8.

        Notes
        -----
        OpenCV 4.11+ требует mask=None в detectAndCompute.
        """
        return self.orb.detectAndCompute(gray, None)  # ← ВОТ ЭТО!

    def track(
            self,
            frame_rgb: np.ndarray,
            timestamp: float,       # зарезервировано для будущих фич
            scale_factor: float = 1.0
    ) -> List | None:
        """
        Трекинг позы для одного кадра.

        Parameters
        ----------
        frame_rgb : np.ndarray
            Кадр в формате RGB, форма [H, W, 3], тип uint8.
        timestamp : float
            Время кадра в секундах (не используется в прототипе).
        scale_factor : float, optional
            Масштаб (в метрах), полученный из MiDaS (mean_depth), по умолчанию 1.0.

        Returns
        -------
        pose_scaled : list | None
            Поза в метрах [x, y, z] или None при потере трекинга.

        Notes
        -----
        Пошаговый алгоритм:

        1. Конвертация в grayscale и извлечение ORB-фич.
        2. Если первый кадр — инициализация в [0,0,0].
        3. Матчинг с предыдущим кадром (BFMatcher + cross-check).
        4. Фильтрация: минимум 20 матчей, топ-100 по расстоянию.
        5. Восстановление Essential Matrix (RANSAC).
        6. Восстановление R и t через recoverPose.
        7. Накопление глобальной позы: R_prev @ R, t_prev + R_prev @ t.
        8. Масштабирование на scale_factor и возврат [x,y,z].
        """
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        kp: List[cv2.KeyPoint]
        kp, desc = self._extract(gray)

        # Недостаточно фич
        if desc is None or len(kp) < 10:
            return None

        # Первый кадр
        if self.last_desc is None:
            self.last_kp = kp
            self.last_desc = desc
            return None  # не добавляем точку

        # Матчинг
        matches: List[cv2.DMatch] = self.matcher.match(self.last_desc, desc)
        if len(matches) < 20:
            return None

        # Топ-100 лучших матчей
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        # Точки для Essential Matrix
        pts1 = np.float32([self.last_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Essential Matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2,
            focal=self.focal,
            pp=(self.cx, self.cy),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        if E is None:
            return None

        # Восстановление позы
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, focal=self.focal)
        if R is None or t is None:
            return None

        # t — в нормированных координатах (пиксели / фокус)
        # t_norm = [Δx/f, Δy/f, Δz/f]
        t_norm = t.flatten()  # [tx, ty, tz]

        baseline = scale_factor / self.focal    # Z / f  → метры на пиксель
        t_m = t_norm * baseline                 # переводим в метры

        # Накопление глобальной позы
        self.R_prev = self.R_prev @ R
        self.t_prev = self.t_prev + (self.R_prev @ t_m.reshape(3, 1))
        pose = self.t_prev.flatten().tolist()

        # Сохранение состояния
        self.trajectory.append(pose)
        self.last_kp = kp
        self.last_desc = desc

        return pose

    def get_trajectory(
            self
    ) -> List:
        """
        Возвращает копию накопленной траектории.

        Returns
        -------
        list
            Список поз [[x, y, z], ...] в метрах.
        """
        return self.trajectory.copy()

    def shutdown(
            self
    ) -> None:
        """
        Освобождение ресурсов и финализация.

        Очищает состояние и выводит статистику.
        """
        self.last_kp = None
        self.last_desc = None
        print(f"[SLAM] Завершено. Траектория: {len(self.trajectory)} точек")
