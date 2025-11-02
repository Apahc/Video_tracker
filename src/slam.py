# src/slam.py
import cv2
import numpy as np
from pathlib import Path
import yaml

class ORBSLAM3Tracker:
    def __init__(self, vocab_path: str, config_path: str):
        self.vocab_path = Path(vocab_path)
        self.config_path = Path(config_path)

        with open(self.config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        cam = cfg['camera']
        orb_cfg = cfg['orb']

        self.focal = float(cam['fx'])
        self.width = int(cam['width'])
        self.height = int(cam['height'])
        self.cx = float(cam.get('cx', self.width / 2))
        self.cy = float(cam.get('cy', self.height / 2))

        self.n_features = int(orb_cfg['n_features'])
        self.scale_factor = float(orb_cfg['scale_factor'])
        self.n_levels = int(orb_cfg['n_levels'])
        self.ini_th_fast = int(orb_cfg['ini_th_fast'])

        # === OpenCV ORB (всегда) ===
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

        # === SLAM ===
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.last_kp = None
        self.last_desc = None
        self.trajectory = []
        self.R_prev = np.eye(3, dtype=np.float32)
        self.t_prev = np.zeros((3, 1), dtype=np.float32)

        print(f"[SLAM] Инициализирован: {self.width}x{self.height}, f={self.focal}")

    def _extract(self, gray):
        """OpenCV 4.11+ требует mask=None"""
        return self.orb.detectAndCompute(gray, None)  # ← ВОТ ЭТО!

    def track(self, frame_rgb: np.ndarray, timestamp: float, scale_factor: float = 1.0):
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        kp, desc = self._extract(gray)

        if desc is None or len(kp) < 10:
            return None

        if self.last_desc is None:
            self.last_kp = kp
            self.last_desc = desc
            self.trajectory.append([0.0, 0.0, 0.0])
            return [0.0, 0.0, 0.0]

        matches = self.matcher.match(self.last_desc, desc)
        if len(matches) < 20:
            return None

        matches = sorted(matches, key=lambda x: x.distance)[:100]

        pts1 = np.float32([self.last_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

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

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, focal=self.focal)

        self.R_prev = self.R_prev @ R
        self.t_prev = self.t_prev + self.R_prev @ t

        pose = [self.t_prev[0,0], self.t_prev[1,0], self.t_prev[2,0]]
        pose_scaled = [p * scale_factor for p in pose]

        self.last_kp = kp
        self.last_desc = desc
        self.trajectory.append(pose_scaled)

        return pose_scaled

    def get_trajectory(self):
        return self.trajectory.copy()

    def shutdown(self):
        self.last_kp = None
        self.last_desc = None
        print(f"[SLAM] Завершено. Траектория: {len(self.trajectory)} точек")