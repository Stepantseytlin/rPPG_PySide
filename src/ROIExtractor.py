from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class ROIExtractor:
    def __init__(self):
        self.options = None
        self.FaceLandmarker = None
        self.indices_options = None
        self.landmarker = None
        self.initialized = 0
    def init(self, model_path: str | None = None, inds: str | dict = "all"):
        if model_path is None:
            model_path =  Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Mediapipe model not found: {model_path}")
        self.options, self.FaceLandmarker = self.initialize_Mediapipe_(str(model_path))
        self.landmarker = self.FaceLandmarker.create_from_options(self.options)
        self.indices_options = inds
        self.initialized = 1

    def initialize_Mediapipe_(self, model_path: str):
        BaseOptions = python.BaseOptions
        FaceLandmarker = vision.FaceLandmarker
        FaceLandmarkerOptions = vision.FaceLandmarkerOptions
        VisionRunningMode = vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
        )
        return options, FaceLandmarker

    def polygon_mask(self, frame_shape, landmarks, indices):
        h, w = frame_shape[:2]
        points = []
        if indices == "all":
            indices = np.arange(len(landmarks))
        for idx in indices:
            lm = landmarks[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append([x, y])

        points = np.array(points, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        return mask

    def extract(self, frame_bgr, timestamp):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp)
        mean_rgb = None
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            if self.indices_options == "all":
                roi_mask = self.polygon_mask(frame_bgr.shape, landmarks, "all")
                mask = cv2.bitwise_or(mask, roi_mask)
            else:
                for roi_indices in self.indices_options.values():
                    roi_mask = self.polygon_mask(frame_bgr.shape, landmarks, roi_indices)
                    mask = cv2.bitwise_or(mask, roi_mask)
            mean_rgb = self.mean_rgb_in_mask(frame_bgr, mask)
        else:   return None,None
        return mean_rgb, cv2.bitwise_or(frame_bgr,np.stack((np.zeros_like(mask),mask,np.zeros_like(mask)),axis=2) )

    def mean_rgb_in_mask(self, frame_bgr, mask):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return cv2.mean(frame_rgb, mask=mask)[:3]
