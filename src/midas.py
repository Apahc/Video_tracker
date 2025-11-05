"""
Модуль для оценки глубины (depth estimation) с использованием модели MiDaS.

Этот модуль реализует предобработку кадров видео и вычисление карты глубины,
которая используется для масштабирования траектории в реальные метры.

Модель MiDaS предоставляет монокулярную оценку глубины, которая нормализуется
до максимума 10 метров (эмпирическое значение для типичных сцен bodycam).
Средняя глубина (mean_depth) используется как scale_factor в SLAM для pixel-to-meter преобразования.

Путь до данного файла: src/midas.py
"""
import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize

from typing import Tuple


class MiDaSEstimator:
    """
    Класс для оценки глубины с использованием предобученной модели MiDaS_small.

    Attributes
    ----------
    device : torch.device
        Устройство для выполнения вычислений (CPU или CUDA).
    model : torch.nn.Module
        Загруженная модель MiDaS_small в режиме оценки.
    transform : torchvision.transforms.Compose
        Набор трансформаций для подготовки входного изображения.
    """
    def __init__(
            self,
            device: str = "cpu"
    ) -> None:
        """
        Инициализация модели MiDaS.

        Parameters
        ----------
        device : str, optional
            Устройство для вычислений ('cpu' или 'cuda'), по умолчанию 'cpu'.

        Notes
        -----
        - Модель загружается из torch.hub с предобученными весами.
        - Переводится в режим ``eval()`` для отключения dropout и batch norm.
        - Трансформации включают конвертацию в тензор и нормализацию
          (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) — стандарт для MiDaS.
        """
        self.device = torch.device(device)
        print(f"[MiDaS] Загрузка на {device}...")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        self.transform = Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    @torch.no_grad()
    def predict(
            self,
            frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Предсказание карты глубины для одного кадра.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Входной кадр в формате BGR (как возвращает ``cv2.VideoCapture().read()``).
            Форма: [H, W, 3], тип: uint8.

        Returns
        -------
        depth_m : np.ndarray
            Карта глубины в метрах, форма: [H, W], тип: float32.
            Значения нормализованы до диапазона [0, 10] метров.
        mean_depth : float
            Средняя глубина по пикселям, где глубина > 0.1 м.
            Используется как масштабный коэффициент для SLAM.

        Notes
        -----
        Алгоритм:

        1. Конвертация BGR -> RGB.
        2. Паддинг изображения до размеров, кратных 32 (stride модели).
        3. Применение трансформаций и добавление batch-дименсии.
        4. Прогон через модель (без градиентов).
        5. Удаление batch и channel дименсий.
        6. Ресайз до оригинального размера с кубической интерполяцией.
        7. Нормализация: depth = depth / max(depth) * 10.0.
        8. Вычисление средней глубины с фильтром шума (глубина > 0.1 м).
        """
        # 1. Конвертация цветового пространства
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = img_rgb.shape[:2]

        # 2. Паддинг до кратности 32 (требование CNN)
        h = ((original_h + 31) // 32) * 32
        w = ((original_w + 31) // 32) * 32
        img_padded = cv2.copyMakeBorder(
            img_rgb,
            0, h - original_h,
            0, w - original_w,
            cv2.BORDER_REFLECT_101  # Отражение для минимизации артефактов
        )

        # 3. Подготовка тензора: [1, 3, H_padded, W_padded]
        input_tensor = self.transform(img_padded).unsqueeze(0).to(self.device)  # [1, 3, H, W]

        # 4. Прогноз модели
        depth = self.model(input_tensor)        # [1, 1, H//32, W//32]

        # 5. Убираем batch и channel
        depth = depth.squeeze(0).squeeze(0)     # [H//32, W//32]

        # 6. Ресайз до оригинального размера
        depth = cv2.resize(
            depth.cpu().numpy(),
            (original_w, original_h),
            interpolation=cv2.INTER_CUBIC
        )

        # 7. Нормализация до 10 метров
        depth_max = depth.max()
        if depth_max > 0:
            depth_m = depth / (depth_max + 1e-6) * 10.0
        else:
            depth_m = depth.copy()

        # 8. Средняя глубина с фильтром шума
        valid_depth = depth_m[depth_m > 0.1]
        mean_depth = float(np.mean(valid_depth)) if valid_depth.size > 0 else 0.0

        return depth_m, mean_depth
