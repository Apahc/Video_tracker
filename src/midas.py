# src/midas.py
import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize

class MiDaSEstimator:
    def __init__(self, device: str = "cpu"):
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
    def predict(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = img_rgb.shape[:2]

        # Приводим к кратному 32
        h = ((original_h + 31) // 32) * 32
        w = ((original_w + 31) // 32) * 32
        img_padded = cv2.copyMakeBorder(
            img_rgb, 0, h - original_h, 0, w - original_w,
            cv2.BORDER_REFLECT_101
        )

        input_tensor = self.transform(img_padded).unsqueeze(0).to(self.device)  # [1, 3, H, W]

        # Прогноз: [1, 1, H//32, W//32]
        depth = self.model(input_tensor)  # ← [1,1,22,40]

        # Убираем batch и channel
        depth = depth.squeeze(0).squeeze(0)  # [H//32, W//32]

        # Ресайзим до оригинального размера
        depth = cv2.resize(
            depth.cpu().numpy(),
            (original_w, original_h),
            interpolation=cv2.INTER_CUBIC
        )

        # Масштабируем до 10 метров
        depth_m = depth / (depth.max() + 1e-6) * 10.0
        mean_depth = float(np.mean(depth_m[depth_m > 0.1]))
        return depth_m, mean_depth