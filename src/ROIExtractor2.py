from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.pose_correction_stream import (
    ROI_NAMES,
    ROIObservation,
    PoseCorrectionStream,
)

class ROIExtractor:
    def __init__(self):
        self.initialized = 0
        self.options = None
        self.FaceLandmarker = None
        self.indices_options = None
        self.landmarker = None
        self.misfps_cnt = 0
    def init(self, model_path: str | None = None, inds: str | dict = "all", fps = 30):
        self.pose_corrector: PoseCorrectionStream | None = None
        self.pose_sequence_id = 0
        self.last_pose_sample = None
        self.fps = np.float64(fps)
        self.pose_corrector = PoseCorrectionStream(fps=fps)
        self.pose_sequence_id = 0
        if model_path is None:
            model_path =  Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Mediapipe model not found: {model_path}")
        self.options, self.FaceLandmarker = self.initialize_Mediapipe_(str(model_path))
        self.landmarker = self.FaceLandmarker.create_from_options(self.options)
        self.indices_options = inds
        self.initialized = 1
        self.ROI_KEYPOINTS_3 = {
    'glabella': [151, 108, 107, 55, 8, 285, 336, 337],
    'left_malar': [126, 100, 118, 117, 116, 123, 147, 187, 205, 203, 129, 209],
    'right_malar': [355, 429, 358, 423, 425, 411, 376, 352, 345, 346, 347, 329]
}

        self.ROI_KEYPOINTS_7 = {
    'glabella': [151, 108, 107, 55, 8, 285, 336, 337],
    'medial_forehead': [10, 109, 108, 151, 337, 338],
    'left_lateral_forehead': [67, 103, 104, 105, 66, 107, 108, 109],
    'right_lateral_forehead': [297, 338, 337, 336, 296, 334, 333, 332],
    'left_malar': [126, 100, 118, 117, 116, 123, 147, 187, 205, 203, 129, 209],
    'right_malar': [355, 429, 358, 423, 425, 411, 376, 352, 345, 346, 347, 329],
    'upper_nasal_dorsum': [8, 193, 122, 196, 197, 419, 351, 417]
}
        self.ROI_KEYPOINTS = self.ROI_KEYPOINTS_7
        if tuple(self.ROI_KEYPOINTS.keys()) != ROI_NAMES:
            raise RuntimeError("Порядок ROI не совпадает с PoseCorrectionStream")
        self.ROI_MARGIN_X = 0.01
        self.ROI_MARGIN_Y = 0.01
    def get_roi_bbox(self, landmarks, img_width, img_height, keypoint_indices):
        """Вычисляет bounding box для ROI с небольшими отступами"""
        points = []
        for idx in keypoint_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                x = int(lm.x * img_width)
                y = int(lm.y * img_height)
                points.append((x, y))
        if len(points) < 3:
            return None
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        width = max_x - min_x
        height = max_y - min_y
        margin_x = int(width * self.ROI_MARGIN_X)
        margin_y = int(height * self.ROI_MARGIN_Y)
        x = max(0, min_x - margin_x)
        y = max(0, min_y - margin_y)
        w = max(1, width + 2 * margin_x)
        h = max(1, height + 2 * margin_y)
        if x + w > img_width:
            w = img_width - x
        if y + h > img_height:
            h = img_height - y
        return (x, y, w, h)
    def draw_landmark_points(self,img_display, landmarks, img_width, img_height, keypoint_indices, roi_name):
        """Отрисовывает жёлтые точки landmark'ов для указанной ROI"""
        for idx in keypoint_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                point_x = int(lm.x * img_width)
                point_y = int(lm.y * img_height)
                cv2.circle(img_display, (point_x, point_y), 4, (0, 255, 255), -1)

    def interpolate_nans_safely(self,arr):
        """
        Безопасная линейная интерполяция пропусков (NaN) в 1D массиве.
        ГАРАНТИРОВАННО возвращает массив ТОЧНО той же длины, что и входной.
        """
        arr = np.array(arr, dtype=np.float64)
        mask = np.isnan(arr)
        
        # Если NaN нет, возвращаем как есть
        if not mask.any():
            return arr
        
        # Если всё состоит из NaN, возвращаем нули
        if mask.all():
            return np.zeros_like(arr)
        
        # Индексы всех элементов массива
        x = np.arange(len(arr))
        # Индексы валидных (не NaN) значений
        xp = np.flatnonzero(~mask)
        # Значения валидных элементов
        fp = arr[~mask]
        
        # np.interp автоматически экстраполирует крайние значения, 
        # что идеально заполняет NaN в начале и конце массива.
        return np.interp(x, xp, fp)
    def initialize_Mediapipe_(self, model_path: str):
        base_options = python.BaseOptions(model_asset_path=model_path)
        base_options.delegate = mp.tasks.BaseOptions.Delegate.CPU

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        try:
            FaceLandmarker = vision.FaceLandmarker.create_from_options(options)
            print("MediaPipe Face Landmarker успешно инициализирован")
        except Exception as e:
            print(f"Ошибка инициализации: {e}")

        

        
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
    def extract_with_pose_correction(self, frame_bgr, timestamp):
        pass


    def extract(self, frame_bgr, timestamp, fps = None):
        if fps is not None:
            if abs(fps - self.fps)>2.0:
                self.misfps_cnt +=1
                if self.misfps_cnt > 10:
                    self.misfps_cnt=0
                    self.fps=fps
                    self.init(fps = fps)
            else:
                self.misfps_cnt = 0
        frame_rgb = frame_bgr#cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb_copy = frame_rgb.copy()
        mean_rgb_scene = np.mean(frame_rgb[0:0+20, 0:0+20],axis=(0,1))
        # print(f"MEAN HERE {mean_rgb_scene, mean_rgb_scene.shape}")
        img_height,img_width, _ =frame_rgb.shape 
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp)
        mean_rgb = None
        roi_rgb = np.full((len(ROI_NAMES), 3), np.nan, dtype=np.float64)
        roi_valid = np.zeros(len(ROI_NAMES), dtype=bool)
        landmarks_xyz = None

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            landmarks_xyz = np.asarray(
                [[lm.x, lm.y, lm.z] for lm in landmarks],
                dtype=np.float64,
            )

            for roi_index, roi_name in enumerate(ROI_NAMES):
                keypoint_indices = self.ROI_KEYPOINTS[roi_name]

                roi_bbox = self.get_roi_bbox(
                    landmarks,
                    img_width,
                    img_height,
                    keypoint_indices,
                )
                if roi_bbox is None:
                    continue

                x, y, w, h = roi_bbox

                cv2.rectangle(
                    frame_rgb_copy,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame_rgb_copy,
                    roi_name,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                self.draw_landmark_points(
                    frame_rgb_copy,
                    landmarks,
                    img_width,
                    img_height,
                    keypoint_indices,
                    roi_name,
                )

                if w <= 0 or h <= 0:
                    continue

                roi_region = frame_rgb[y:y + h, x:x + w]
                if roi_region.size == 0:
                    continue

                # frame_rgb уже имеет порядок каналов RGB.
                mean_rgb = np.mean(
                    roi_region,
                    axis=(0, 1),
                    dtype=np.float64,
                )

                if np.all(np.isfinite(mean_rgb)):
                    roi_rgb[roi_index] = mean_rgb
                    roi_valid[roi_index] = True
        if self.pose_corrector is None:
            raise RuntimeError("ROIExtractor.init() не был вызван")

        sample = self.pose_corrector.update(
            ROIObservation(
                sequence_id=self.pose_sequence_id,
                # timestamp MediaPipe передаётся в миллисекундах.
                timestamp_seconds=float(timestamp) / 1000.0,
                roi_rgb=roi_rgb,
                landmarks_xyz=landmarks_xyz,
                roi_valid=roi_valid,
            )
        )

        self.pose_sequence_id += 1
        self.last_pose_sample = sample

        # При отсутствии ROI sample.rgb является fallback и в FIFO идти не должен.
        corrected_rgb = sample.rgb.copy() if sample.valid else None

        return corrected_rgb, frame_rgb_copy



    def extract2(self, frame_bgr, timestamp):
        frame_rgb = frame_bgr#cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb_copy = frame_rgb.copy()
        mean_rgb_scene = np.mean(frame_rgb[0:0+20, 0:0+20],axis=(0,1))
        # print(f"MEAN HERE {mean_rgb_scene, mean_rgb_scene.shape}")
        img_height,img_width, _ =frame_rgb.shape 
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp)
        mean_rgb = None

        if result.face_landmarks:
            roi_r_values = {name: 0 for name in self.ROI_KEYPOINTS.keys()}
            roi_g_values = {name: 0 for name in self.ROI_KEYPOINTS.keys()}
            roi_b_values = {name: 0 for name in self.ROI_KEYPOINTS.keys()}
            for roi_name, keypoint_indices in self.ROI_KEYPOINTS.items():
                            roi_bbox = self.get_roi_bbox(result.face_landmarks[0], img_width, img_height, keypoint_indices)
                            
                            if roi_bbox is not None:
                                (x, y, w, h) = roi_bbox
                                
                                # Отрисовка ROI и landmark'ов
                                
                                cv2.rectangle(frame_rgb_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(frame_rgb_copy, roi_name, (x, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                self.draw_landmark_points(frame_rgb_copy, result.face_landmarks[0], img_width, img_height, 
                                                    keypoint_indices, roi_name)
                                
                                if w > 0 and h > 0:
                                    roi_region = frame_rgb[y:y+h, x:x+w]
                                    # BGR порядок в OpenCV: 0=B, 1=G, 2=R
                                    roi_r_values[roi_name]=np.mean(roi_region[:, :, 2])
                                    roi_g_values[roi_name]=np.mean(roi_region[:, :, 1])
                                    roi_b_values[roi_name]=np.mean(roi_region[:, :, 0])
                                else:
                                    # ROI найдена, но имеет нулевой размер -> NaN
                                    roi_r_values[roi_name]=np.nan
                                    roi_g_values[roi_name]=np.nan
                                    roi_b_values[roi_name]=np.nan
                            else:
                                # ROI не найдена (например, скрыта при повороте головы) -> NaN
                                roi_r_values[roi_name]=np.nan
                                roi_g_values[roi_name]=np.nan
                                roi_b_values[roi_name]=np.nan
        else:   return None,None
        all_R_arr = np.array([roi_r_values[name] for name in self.ROI_KEYPOINTS.keys()], dtype=np.float64)
        all_G_arr = np.array([roi_g_values[name] for name in self.ROI_KEYPOINTS.keys()], dtype=np.float64)
        all_B_arr = np.array([roi_b_values[name] for name in self.ROI_KEYPOINTS.keys()], dtype=np.float64)

        R_avg = np.nanmean(all_R_arr, axis=0)
        G_avg = np.nanmean(all_G_arr, axis=0)
        B_avg = np.nanmean(all_B_arr, axis=0)
        mean_rgb = np.column_stack([R_avg, G_avg, B_avg])
        # print(f"WHILE ROI IS {mean_rgb}")
        mean_scene = mean_rgb_scene #- mean_rgb
        out = mean_rgb.reshape(-1) #- mean_scene.reshape(-1)
        # print(f"RETURNING {out}")
        # print("MEANRGB SHAPE ",mean_rgb.shape)
        return out, frame_rgb_copy

    def mean_rgb_in_mask(self, frame_bgr, mask):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return cv2.mean(frame_rgb, mask=mask)[:3]