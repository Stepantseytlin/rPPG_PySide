# ============================================================================
# pose_correction_ws.py
# МОДУЛЬ: КОРРЕКЦИИ ROI ПО ПОЗЕ ГОЛОВЫ (ВЕРСИЯ ДЛЯ WEBSOCKET ИНТЕГРАЦИИ)
# ============================================================================

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy.linalg import block_diag
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis

@dataclass
class PoseInfo:
    """Информация о позе."""
    yaw: np.ndarray = field(default_factory=lambda: np.array([]))
    pitch: np.ndarray = field(default_factory=lambda: np.array([]))
    roll: np.ndarray = field(default_factory=lambda: np.array([]))
    valid_frames: np.ndarray = field(default_factory=lambda: np.array([]))
    valid_ratio: float = 0.0
    roi_weights: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class CorrectedSignals:
    """Скорректированные сигналы."""
    R: np.ndarray = field(default_factory=lambda: np.array([]))
    G: np.ndarray = field(default_factory=lambda: np.array([]))
    B: np.ndarray = field(default_factory=lambda: np.array([]))
    valid_frames: np.ndarray = field(default_factory=lambda: np.array([]))
    roi_weights: np.ndarray = field(default_factory=lambda: np.array([]))

class PoseConstrainedKalmanFilter:
    """PCKF для стабилизации landmark'ов (Liu et al., 2021)."""
   
    def __init__(self, num_landmarks=14, fps=25.0, process_noise=0.001, measurement_noise=0.01):
        self.num_landmarks = num_landmarks
        self.fps = fps
        self.state = np.zeros(num_landmarks * 4)
        self.P = np.eye(num_landmarks * 4) * 0.1
       
        dt = 1.0 / fps
        A_single = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.A = block_diag(*[A_single] * num_landmarks)
       
        H_single = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.H = block_diag(*[H_single] * num_landmarks)
       
        self.Q = np.eye(num_landmarks * 4) * process_noise
        self.R = np.eye(num_landmarks * 2) * measurement_noise
       
    def update(self, measured_landmarks):
        """Коррекция координат landmark'ов."""
        self.state = self.A @ self.state
        self.P = self.A @ self.P @ self.A.T + self.Q
       
        z = measured_landmarks.flatten()
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K_gain = self.P @ self.H.T @ np.linalg.inv(S)
       
        self.state = self.state + K_gain @ y
        self.P = (np.eye(len(self.state)) - K_gain @ self.H) @ self.P
       
        return self.state.reshape(-1, 4)[:, :2]

def estimate_head_pose_geometric(landmarks_14: np.ndarray) -> Tuple[float, float, float]:
    """Оценка позы через 14 landmark'ов."""
    if landmarks_14.shape[0] < 14:
        return np.nan, np.nan, np.nan
   
    nose_tip = landmarks_14[8, :2]
    left_eye_inner = landmarks_14[4, :2]
    right_eye_inner = landmarks_14[5, :2]
   
    face_center = (left_eye_inner + right_eye_inner) / 2
    dx = nose_tip[0] - face_center[0]
    eye_distance = np.linalg.norm(left_eye_inner - right_eye_inner)
   
    if eye_distance > 0.01:
        yaw = np.degrees(np.arctan2(dx, eye_distance * 0.5 + 0.01))
    else:
        yaw = 0.0
   
    chin = landmarks_14[13, 1]
    nose_bridge = landmarks_14[9, 1]
    nose_tip_y = landmarks_14[8, 1]
    left_eye_y = landmarks_14[4, 1]
    right_eye_y = landmarks_14[5, 1]
    eye_y = (left_eye_y + right_eye_y) / 2
   
    face_height = chin - nose_bridge
    if abs(face_height) > 0.02:
        nose_position = (nose_tip_y - eye_y) / face_height
        pitch = (nose_position - 0.5) * 100
    else:
        pitch = 0.0
   
    left_eye_outer = landmarks_14[6, :]
    right_eye_outer = landmarks_14[7, :]
    dx_roll = right_eye_outer[0] - left_eye_outer[0]
    dy_roll = right_eye_outer[1] - left_eye_outer[1]
   
    if abs(dx_roll) > 1e-6:
        roll = np.degrees(np.arctan2(dy_roll, dx_roll))
    else:
        roll = 0.0
   
    return float(np.clip(yaw, -90, 90)), float(np.clip(pitch, -60, 60)), float(np.clip(roll, -60, 60))

def compute_snr_weight(g_signal: np.ndarray, fps: float = 25.0, snr_threshold: float = 0.293) -> Tuple[float, bool]:
    """
    Вычисление веса ROI на основе SNR в диапазоне ЧСС с Butterworth фильтрацией.
   
    - Butterworth bandpass фильтр (0.667-4 Гц) перед вычислением SNR
    - Адаптивный порог SNR < 0.293 для исключения плохих ROI
   
    Args:
        g_signal: сигнал ROI
        fps: частота кадров
        snr_threshold: порог SNR 
   
    Returns:
        snr_weight: вес от 0.01 до 1.0 (выше = лучше качество)
        is_below_threshold: флаг, что SNR ниже порога (для диагностики)
    """
    # ===BUTTERWORTH ФИЛЬТРАЦИЯ ===
    nyquist = fps / 2.0
    lowcut = 0.667 / nyquist
    highcut = 4.0 / nyquist
   
    if lowcut >= highcut or lowcut <= 0 or highcut >= 1:
        filtered_signal = g_signal
    else:
        try:
            b, a = butter(4, [lowcut, highcut], btype='band')
            filtered_signal = filtfilt(b, a, g_signal)
        except Exception:
            filtered_signal = g_signal
   
    # Вычисление SNR от фильтрованного сигнала
    f, Pxx = signal.welch(filtered_signal, fs=fps, nperseg=min(125, len(filtered_signal)))
   
    mask_hr = (f >= 0.667) & (f <= 4.0)
    power_hr = np.sum(Pxx[mask_hr])
    power_total = np.sum(Pxx)
   
    if power_total > 1e-10:
        snr = power_hr / (power_total - power_hr + 1e-10)
       
        # === АДАПТИВНЫЙ ПОРОГ SNR ===
        if snr < snr_threshold:
            snr_weight = 0.01
            is_below_threshold = True
        else:
            snr_weight = np.clip(snr * 2.0, 0.1, 1.0)
            is_below_threshold = False
    else:
        snr_weight = 0.01
        is_below_threshold = True
   
    return snr_weight, is_below_threshold

def compute_kurtosis_weight(g_signal: np.ndarray, fps: float = 25.0) -> float:
    """
    Вычисление веса ROI на основе Kurtosis (эксцесса) сигнала.
   
    - Kurtosis эффективнее SNR для различения acceptable/unfit сигналов
    - Оптимальный диапазон kurtosis для rPPG: -0.5 до 1.5
   
    Защита от NaN и константных сигналов
    - Предотвращает RuntimeWarning: Precision loss due to catastrophic cancellation
   
    Returns:
        kurtosis_weight: вес от 0.1 до 1.0 (выше = лучше качество)
    """
    # === KURTOSIS ВЗВЕШИВАНИЕ ===
    if len(g_signal) < 10:
        return 0.5
   
    # === ЗАЩИТА ОТ NAN И КОНСТАНТНЫХ СИГНАЛОВ  ===
    # Предотвращает RuntimeWarning: Precision loss due to catastrophic cancellation
    if np.isnan(g_signal).any() or np.std(g_signal) < 1e-5:
        return 0.5  # Нейтральный вес для проблемных сигналов
    
    try:
        kurt = kurtosis(g_signal, fisher=True)
       
        kurt_optimal_min = -0.5
        kurt_optimal_max = 1.5
       
        if kurt_optimal_min <= kurt <= kurt_optimal_max:
            kurtosis_weight = 1.0
        else:
            if kurt < kurt_optimal_min:
                distance = kurt_optimal_min - kurt
            else:
                distance = kurt - kurt_optimal_max
           
            kurtosis_weight = np.clip(1.0 - distance / 3.0, 0.1, 1.0)
    except Exception:
        kurtosis_weight = 0.5
   
    return kurtosis_weight

def compute_surface_normal_weight(landmarks_3d: np.ndarray, roi_indices: list) -> float:
    """
    Вычисление веса ROI на основе векторов нормали поверхности (3D landmark'ы).
   
    - Surface normal vectors влияют на соотношение specular/diffuse отражения
    - Только diffuse отражение несёт физиологическую информацию
    - Оптимальный угол отражения: ~60° (Fan et al., 2024)
    
   Асимметричная функция штрафования
    - Пик качества около 45-60° (оптимальный баланс diffuse/specular)
    - Резкий спад для углов >75° (профиль, где диффузное отражение минимально)
    - Предотвращает ошибку, когда профиль получал вес ВЫШЕ фронтального вида
   
    Args:
        landmarks_3d: 3D landmark'ы MediaPipe (468 точек, включая Z-координату)
        roi_indices: индексы landmark'ов для данной ROI
   
    Returns:
        surface_weight: вес от 0.05 до 1.0 (выше = лучше угол отражения)
    """
    # ===  SURFACE NORMAL VECTORS  ===
    if len(roi_indices) < 3 or landmarks_3d.shape[0] < max(roi_indices):
        return 0.5  # Default weight if not enough points
   
    try:
        # Получаем 3D координаты landmark'ов для ROI
        roi_points = landmarks_3d[roi_indices, :3]  # x, y, z
       
        # Вычисляем нормаль поверхности через PCA
        centroid = np.mean(roi_points, axis=0)
        centered_points = roi_points - centroid
       
        # Ковариационная матрица
        cov_matrix = np.cov(centered_points.T)
       
        # Собственные векторы (нормали)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
       
        # Наименьший собственный вектор = нормаль к поверхности
        surface_normal = eigenvectors[:, 0]
       
        # Направление камеры 
        camera_direction = np.array([0, 0, 1])
       
        # Угол между нормалью поверхности и направлением камеры
        dot_product = np.dot(surface_normal, camera_direction)
        magnitude_normal = np.linalg.norm(surface_normal)
        magnitude_camera = np.linalg.norm(camera_direction)
       
        if magnitude_normal * magnitude_camera > 1e-10:
            cos_angle = dot_product / (magnitude_normal * magnitude_camera)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_degrees = np.degrees(np.arccos(np.abs(cos_angle)))
        else:
            angle_degrees = 90.0
       
        # ===  АСИММЕТРИЧНОЕ ШТРАФОВАНИЕ ===
        # Пик качества около 45-60° (оптимальный баланс diffuse/specular)
        # Резкий спад для углов >75° (профиль, где диффузное отражение минимально)
        
        if angle_degrees <= 75.0:
            # Мягкий спад от 50° к краям (0° и 75°)
            # Пик около 50° (среднее между 45° и 60°)
            surface_weight = np.clip(1.0 - (abs(angle_degrees - 50.0) / 50.0), 0.3, 1.0)
        else:
            # Резкое падение для профильных видов (>75°), где диффузное отражение минимально
            surface_weight = np.clip(1.0 - (angle_degrees - 75.0) / 15.0, 0.05, 0.3)
       
        return float(surface_weight)
   
    except Exception:
        return 0.5  # Default weight on error

def compute_local_motion_weight(landmarks: Dict[str, np.ndarray], 
                                roi_indices: list, 
                                current_frame: int, 
                                global_disp: float,
                                alpha: float = 15.0) -> float:
    """
    Sparse Optical Flow (детекция локального движения внутри ROI).
    
    Вычисляет среднее смещение landmark'ов внутри конкретной ROI между текущим 
    и предыдущим кадром. Вычитает глобальное смещение головы, чтобы изолировать 
    локальные артефакты (мимика, моргание, сбой трекинга).
    
    Args:
        landmarks: словарь landmark'ов для каждого кадра
        roi_indices: индексы landmark'ов для данной ROI
        current_frame: номер текущего кадра (0-indexed)
        global_disp: глобальное смещение головы (для компенсации)
        alpha: коэффициент чувствительности (чем выше, тем строже штраф за движение)
    
    Returns:
        motion_weight: вес от 0.01 до 1.0 (выше = меньше локального движения)
    """
    if current_frame < 1:
        return 1.0
    
    frame_key_curr = f'frame_{current_frame + 1}'
    frame_key_prev = f'frame_{current_frame}'
    
    if frame_key_curr not in landmarks or frame_key_prev not in landmarks:
        return 1.0
    
    lm_curr = landmarks[frame_key_curr]
    lm_prev = landmarks[frame_key_prev]
    
    if lm_curr.shape[0] < max(roi_indices) or lm_prev.shape[0] < max(roi_indices):
        return 1.0
    
    # Берём только x, y координаты landmark'ов внутри ROI
    points_curr = lm_curr[roi_indices, :2]
    points_prev = lm_prev[roi_indices, :2]
    
    # Вычисляем евклидово расстояние для каждой точки (вектор оптического потока)
    displacements = np.linalg.norm(points_curr - points_prev, axis=1)
    
    # Среднее локальное смещение внутри ROI
    local_disp = np.mean(displacements)
    
    # Изолируем локальное движение (вычитаем глобальное движение головы)
    # Если local_disp <= global_disp, значит ROI движется вместе с головой (это нормально)
    relative_motion = max(0.0, local_disp - global_disp)
    
    # Экспоненциальное затухание веса при наличии локального движения
    motion_weight = np.exp(-alpha * relative_motion)
    motion_weight = np.clip(motion_weight, 0.01, 1.0)
    
    return float(motion_weight)

def compute_motion_robust_weights(landmarks: Dict[str, np.ndarray],
                                  rgb_signals: Dict[str, np.ndarray],
                                  num_frames: int,
                                  num_roi: int = 7,
                                  fps: float = 25.0,
                                  snr_threshold: float = 0.293) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Вычисление весов ROI на основе позы + SNR + Kurtosis + Surface Normal + Motion).
    """
    roi_names = ['glabella', 'medial_forehead', 'left_lateral_forehead',
                 'right_lateral_forehead', 'left_malar', 'right_malar', 'upper_nasal_dorsum']
   
    # Индексы landmark'ов для каждой ROI (для Surface Normal)
    roi_landmark_indices = {
        'glabella': [151, 108, 107, 55, 8, 285, 336, 337],
        'medial_forehead': [10, 109, 108, 151, 337, 338],
        'left_lateral_forehead': [67, 103, 104, 105, 66, 107, 108, 109],
        'right_lateral_forehead': [297, 338, 337, 336, 296, 334, 333, 332],
        'left_malar': [126, 100, 118, 117, 116, 123, 147, 187, 205, 203, 129, 209],
        'right_malar': [355, 429, 358, 423, 425, 411, 376, 352, 345, 346, 347, 329],
        'upper_nasal_dorsum': [8, 193, 122, 196, 197, 419, 351, 417]
    }
   
    roi_weights = np.zeros((num_frames, num_roi))
    valid_frames = np.zeros(num_frames, dtype=bool)
    yaw = np.zeros(num_frames)
    pitch = np.zeros(num_frames)
    roll = np.zeros(num_frames)
    snr_weights = np.zeros((num_frames, num_roi))
    kurtosis_weights = np.zeros((num_frames, num_roi))
    snr_below_threshold = np.zeros((num_frames, num_roi), dtype=bool)
    surface_weights = np.zeros((num_frames, num_roi))
    motion_weights = np.zeros((num_frames, num_roi)) # ЭТАП 6
   
    pckf = PoseConstrainedKalmanFilter(num_landmarks=14, fps=fps)
   
    lm_14_indices = [70, 63, 105, 107, 33, 263, 336, 296, 1, 6, 197, 0, 17, 152]
    key_indices = [33, 263, 1, 6, 152]
   
    prev_lm = None
   
    for i in range(num_frames):
        frame_key = f'frame_{i+1}'
       
        if frame_key in landmarks:
            lm = landmarks[frame_key]
           
            if lm.shape[0] >= max(lm_14_indices):
                lm_14 = lm[lm_14_indices, :2]
                lm_14_corrected = pckf.update(lm_14)
                yaw[i], pitch[i], roll[i] = estimate_head_pose_geometric(lm_14_corrected)
               
                if prev_lm is not None and prev_lm.shape[0] >= max(key_indices):
                    displacement = np.mean([np.linalg.norm(lm[idx, :2] - prev_lm[idx, :2]) for idx in key_indices])
                else:
                    displacement = 0.0
               
                prev_lm = lm.copy()
               
                if (not np.isnan(yaw[i]) and not np.isnan(pitch[i]) and not np.isnan(roll[i]) and
                    abs(yaw[i]) < 90 and abs(pitch[i]) < 60 and abs(roll[i]) < 60):
                    valid_frames[i] = True
                   
                    w = np.ones(num_roi)
                   
                    # Фактор смещения
                    disp_factor = np.maximum(0, 1 - displacement / 0.2)
                   
                    # Фактор поворота
                    yaw_factor = np.maximum(0, 1 - abs(yaw[i]) / 75)
                    pitch_factor = np.maximum(0, 1 - abs(pitch[i]) / 50)
                    rotation_factor = np.minimum(yaw_factor, pitch_factor)
                   
                    # Индивидуальные веса при повороте
                    if yaw[i] < -15:
                        w[3] *= (1 + yaw[i] / 45)
                        w[5] *= (1 + yaw[i] / 45)
                    elif yaw[i] > 15:
                        w[2] *= (1 - yaw[i] / 45)
                        w[4] *= (1 - yaw[i] / 45)
                   
                    if pitch[i] > 10:
                        w[1] *= (1 - pitch[i] / 70)
                        w[2] *= (1 - pitch[i] / 70)
                        w[3] *= (1 - pitch[i] / 70)
                   
                    w = w * disp_factor * rotation_factor
                    w = np.maximum(w, 0.05)
                   
                    # ГИБРИДНОЕ ИСКЛЮЧЕНИЕ ЩЁК (вес = 0.05)
                    if abs(yaw[i]) > 30:
                        if yaw[i] > 30:
                            w[2] = 0.05
                            w[4] = 0.05
                        else:
                            w[3] = 0.05
                            w[5] = 0.05
                   
                    # === MULTI-SQI + ВЗВЕШИВАНИЕ ===
                    for roi_idx in range(num_roi):
                        roi_name = roi_names[roi_idx]
                        roi_g = rgb_signals[f'{roi_name}_G']
                        segment_start = max(0, i - 62)
                        segment_end = min(len(roi_g), i + 63)
                        segment = roi_g[segment_start:segment_end]
                       
                        # SNR вес
                        snr_weight, is_below = compute_snr_weight(segment, fps=fps, snr_threshold=snr_threshold)
                        snr_weights[i, roi_idx] = snr_weight
                        snr_below_threshold[i, roi_idx] = is_below
                       
                        # Kurtosis вес — с защитой от NaN и константных сигналов
                        kurtosis_weights[i, roi_idx] = compute_kurtosis_weight(segment, fps=fps)
                       
                        # === Surface Normal вес ===
                        if lm.shape[0] >= 468:
                            surface_weight = compute_surface_normal_weight(
                                lm, roi_landmark_indices[roi_name]
                            )
                        else:
                            surface_weight = 0.5
                        surface_weights[i, roi_idx] = surface_weight
                        
                        # === Optical Flow (локальное движение) ===
                        m_weight = compute_local_motion_weight(
                            landmarks, 
                            roi_landmark_indices[roi_name], 
                            i, 
                            displacement, # глобальное смещение для компенсации
                            alpha=15.0    # чувствительность к локальному движению
                        )
                        motion_weights[i, roi_idx] = m_weight
                   
                    # ГИБРИД: поза × SNR × Kurtosis × Surface Normal × Motion
                    w_snr = snr_weights[i, :]
                    w_kurt = kurtosis_weights[i, :]
                    w_surface = surface_weights[i, :]
                    w_motion = motion_weights[i, :]
                    
                    w = w * w_snr * w_kurt * w_surface * w_motion
                   
                    w = w / np.sum(w)
                    roi_weights[i, :] = w
                else:
                    roi_weights[i, :] = np.ones(num_roi) / num_roi
            else:
                yaw[i] = np.nan
                pitch[i] = np.nan
                roll[i] = np.nan
                roi_weights[i, :] = np.ones(num_roi) / num_roi
        else:
            yaw[i] = np.nan
            pitch[i] = np.nan
            roll[i] = np.nan
            roi_weights[i, :] = np.ones(num_roi) / num_roi
   
    # Сглаживание окном 3 кадра
    print("    Сглаживание весов окном 3 кадра...")
    window_size = 3
    for roi_idx in range(num_roi):
        roi_weights[:, roi_idx] = np.convolve(roi_weights[:, roi_idx], np.ones(window_size)/window_size, mode='same')
   
    for i in range(num_frames):
        roi_weights[i, :] = roi_weights[i, :] / np.sum(roi_weights[i, :])
   
    return roi_weights, valid_frames, yaw, pitch, roll, snr_weights, kurtosis_weights, snr_below_threshold, surface_weights, motion_weights

def correct_pose(rgb_signals: Dict[str, np.ndarray], landmarks: Dict[str, np.ndarray],
                 fps: float, config: Optional[Dict] = None) -> Tuple[CorrectedSignals, PoseInfo]:
    """Коррекция ROI по позе."""
    default_config = {
        'use_pckf': True,
        'use_pose_weighting': True,
        'use_snr_weighting': True,
        'use_kurtosis_weighting': True,
        'use_surface_normal_weighting': True,
        'use_motion_weighting': True, # ЭТАП 6
        'snr_threshold': 0.293,
        'yaw_exclude_threshold': 30,
        'blend_ratio': 0.7,
    }
   
    if config is None:
        config = default_config
    else:
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
   
    roi_names = ['glabella', 'medial_forehead', 'left_lateral_forehead',
                 'right_lateral_forehead', 'left_malar', 'right_malar', 'upper_nasal_dorsum']
   
    lengths = [len(rgb_signals[f'{roi}_R']) for roi in roi_names]
    min_len = min(lengths)
    num_frames = min_len
    num_roi = len(roi_names)
   
    print(f"  Обработка {num_frames} кадров...")
    print(f"  PCKF: {'ВКЛ' if config['use_pckf'] else 'ВЫКЛ'}")
    print(f"  Взвешивание по позе: {'ВКЛ' if config['use_pose_weighting'] else 'ВЫКЛ'}")
    print(f"  SNR взвешивание: {'ВКЛ' if config['use_snr_weighting'] else 'ВЫКЛ'} ")
    print(f"  Адаптивный порог SNR: {config['snr_threshold']} ")
    print(f"  Kurtosis взвешивание: {'ВКЛ' if config['use_kurtosis_weighting'] else 'ВЫКЛ'} ")
    print(f"  Surface Normal взвешивание: {'ВКЛ' if config['use_surface_normal_weighting'] else 'ВЫКЛ'} ")
    print(f"  Откат агрессивной нормализации ROI")
    print(f"  Удалён High-Pass фильтр 0.5 Гц")
    print(f"  Удалено клиппирование ±3σ")
    print(f"  Защита Kurtosis от NaN и константных сигналов")
    print(f"  Motion (Optical Flow) взвешивание: {'ВКЛ' if config['use_motion_weighting'] else 'ВЫКЛ'} ")
    print(f"  Исключение ROI при yaw > {config['yaw_exclude_threshold']}° (вес=0.05)")
    print(f"  Сглаживание весов: окно 3 кадра")
    print(f"  Блендинг: {config['blend_ratio']*100:.0f}% коррекция + {(1-config['blend_ratio'])*100:.0f}% оригинал")
   
    print("\n  Вычисление весов ROI...")
   
    if config['use_pckf']:
        roi_weights, valid_frames, yaw, pitch, roll, snr_weights, kurtosis_weights, snr_below_threshold, surface_weights, motion_weights = compute_motion_robust_weights(
            landmarks, rgb_signals, num_frames, num_roi, fps, config['snr_threshold']
        )
    else:
        roi_weights = np.ones((num_frames, num_roi)) / num_roi
        valid_frames = np.ones(num_frames, dtype=bool)
        yaw = np.zeros(num_frames)
        pitch = np.zeros(num_frames)
        roll = np.zeros(num_frames)
        snr_weights = np.ones((num_frames, num_roi))
        kurtosis_weights = np.ones((num_frames, num_roi))
        snr_below_threshold = np.zeros((num_frames, num_roi), dtype=bool)
        surface_weights = np.ones((num_frames, num_roi))
        motion_weights = np.ones((num_frames, num_roi))
   
    valid_count = np.sum(valid_frames)
    valid_ratio = valid_count / num_frames * 100
    print(f"    Валидные кадры: {valid_count} из {num_frames} ({valid_ratio:.1f}%)")
   
    print(f"\n  Взвешенное усреднение {num_roi} ROI...")
   
    R_matrix = np.column_stack([rgb_signals[f'{roi}_R'][:min_len] for roi in roi_names])
    G_matrix = np.column_stack([rgb_signals[f'{roi}_G'][:min_len] for roi in roi_names])
    B_matrix = np.column_stack([rgb_signals[f'{roi}_B'][:min_len] for roi in roi_names])
   
    original_R = np.mean(R_matrix, axis=1)
    original_G = np.mean(G_matrix, axis=1)
    original_B = np.mean(B_matrix, axis=1)
   
    corrected_R = np.zeros(num_frames)
    corrected_G = np.zeros(num_frames)
    corrected_B = np.zeros(num_frames)
   
    for i in range(num_frames):
        w = roi_weights[i, :]
        corrected_R[i] = np.sum(w * R_matrix[i, :])
        corrected_G[i] = np.sum(w * G_matrix[i, :])
        corrected_B[i] = np.sum(w * B_matrix[i, :])
       
    blend = config['blend_ratio']
    corrected_R = blend * corrected_R + (1 - blend) * original_R
    corrected_G = blend * corrected_G + (1 - blend) * original_G
    corrected_B = blend * corrected_B + (1 - blend) * original_B
   
    print("\n  Статистика позы:")
    valid_yaw = yaw[valid_frames]
    valid_pitch = pitch[valid_frames]
    valid_roll = roll[valid_frames]
   
    if len(valid_yaw) > 0:
        print(f"    Yaw:   {np.mean(valid_yaw):.1f} ± {np.std(valid_yaw):.1f}°")
        print(f"    Pitch: {np.mean(valid_pitch):.1f} ± {np.std(valid_pitch):.1f}°")
        print(f"    Roll:  {np.mean(valid_roll):.1f} ± {np.std(valid_roll):.1f}°")
   
    print(f"    Валидные кадры: {valid_ratio:.1f}%")
    print(f"    Средний вес ROI: {np.mean(roi_weights[valid_frames]):.3f}")
   
    # Диагностика
    print("\n" + "="*70)
    print("ДИАГНОСТИКА ВЕСОВ ROI")
    print("="*70)
    print(f"\nСредний вес ROI: {np.mean(roi_weights[valid_frames]):.6f}")
    print(f"Стандартное отклонение: {np.std(roi_weights[valid_frames]):.6f}")
    print(f"Мин вес: {np.min(roi_weights[valid_frames]):.6f}")
    print(f"Макс вес: {np.max(roi_weights[valid_frames]):.6f}")
   
    if np.std(roi_weights[valid_frames]) > 0.03:
        print(f"\n Веса ХОРОШО дифференцированы (std={np.std(roi_weights[valid_frames]):.4f})")
    else:
        print(f"\n Веса дифференцированы слабо (std={np.std(roi_weights[valid_frames]):.4f})")
   
    print("\n" + "-"*70)
    print("СРЕДНИЕ ВЕСА ДЛЯ КАЖДОГО ROI:")
    print("-"*70)
    for i, name in enumerate(roi_names):
        mean_weight = np.mean(roi_weights[valid_frames, i])
        std_weight = np.std(roi_weights[valid_frames, i])
        print(f"  {i}. {name:25s}: {mean_weight:.4f} ± {std_weight:.4f}")
    print("="*70)
   
    # Диагностика Multi-SQI + Motion
    print("\n" + "-"*70)
    print("ДИАГНОСТИКА MULTI-SQI + Motion")
    print("-"*70)
    print(f"Средний SNR вес: {np.mean(snr_weights[valid_frames]):.4f} ± {np.std(snr_weights[valid_frames]):.4f}")
    print(f"Средний Kurtosis вес: {np.mean(kurtosis_weights[valid_frames]):.4f} ± {np.std(kurtosis_weights[valid_frames]):.4f}")
    print(f"Средний Surface Normal вес: {np.mean(surface_weights[valid_frames]):.4f} ± {np.std(surface_weights[valid_frames]):.4f}")
    print(f"Средний Motion (Optical Flow) вес: {np.mean(motion_weights[valid_frames]):.4f} ± {np.std(motion_weights[valid_frames]):.4f}")
    print("="*70)
   
    # Диагностика адаптивного порога
    total_snr_measurements = np.sum(valid_frames) * num_roi
    snr_below_count = np.sum(snr_below_threshold[valid_frames])
    snr_below_ratio = snr_below_count / total_snr_measurements * 100
    print(f"\nАдаптивный порог SNR < {config['snr_threshold']}:")
    print(f"  ROI ниже порога: {snr_below_count} из {total_snr_measurements} ({snr_below_ratio:.1f}%)")
    print(f"  ROI выше порога: {total_snr_measurements - snr_below_count} ({100 - snr_below_ratio:.1f}%)")
    print("="*70)
   
    print(f"\n  Итоговый диапазон (СЫРЫЕ для MATLAB):")
    print(f"    R: [{corrected_R.min():.2f}, {corrected_R.max():.2f}]")
    print(f"    G: [{corrected_G.min():.2f}, {corrected_G.max():.2f}]")
    print(f"    B: [{corrected_B.min():.2f}, {corrected_B.max():.2f}]")
   
    pose_info = PoseInfo(yaw=yaw, pitch=pitch, roll=roll, valid_frames=valid_frames,
                         valid_ratio=valid_ratio, roi_weights=roi_weights)
   
    corrected_signals = CorrectedSignals(R=corrected_R, G=corrected_G, B=corrected_B,
                                         valid_frames=valid_frames, roi_weights=roi_weights)
   
    print("\n   Коррекция завершена ")
    print("="*70)
   
    return corrected_signals, pose_info, snr_weights, kurtosis_weights, surface_weights, motion_weights

# ============================================================================
# ФУНКЦИЯ ДЛЯ WEBSOCKET ИНТЕГРАЦИИ
# ============================================================================
def extract_diagnostics_for_websocket(
    pose_info: PoseInfo,
    corrected_signals: CorrectedSignals,
    snr_weights: np.ndarray,
    kurtosis_weights: np.ndarray,
    surface_weights: np.ndarray,
    motion_weights: np.ndarray,
    roi_names: list
) -> Dict:
    """
    Извлекает все метаданные из результатов коррекции и возвращает словарь,
    готовый для отправки через WebSocket.
    
    Args:
        pose_info: информация о позе головы
        corrected_signals: скорректированные RGB сигналы
        snr_weights: веса SNR для каждой ROI
        kurtosis_weights: веса Kurtosis для каждой ROI
        surface_weights: веса Surface Normal для каждой ROI
        motion_weights: веса Motion (Optical Flow) для каждой ROI
        roi_names: список названий ROI
    
    Returns:
        Словарь с метаданными, готовый для JSON-сериализации
    """
    # Вычисляем средние значения по валидным кадрам
    valid_frames = pose_info.valid_frames
    
    # Средние веса для каждой ROI
    roi_weights_dict = {}
    for i, name in enumerate(roi_names):
        roi_weights_dict[name] = {
            "mean_weight": float(np.mean(pose_info.roi_weights[valid_frames, i])),
            "std_weight": float(np.std(pose_info.roi_weights[valid_frames, i])),
            "snr_weight": float(np.mean(snr_weights[valid_frames, i])),
            "kurtosis_weight": float(np.mean(kurtosis_weights[valid_frames, i])),
            "surface_weight": float(np.mean(surface_weights[valid_frames, i])),
            "motion_weight": float(np.mean(motion_weights[valid_frames, i]))
        }
    
    # Статистика позы
    pose_stats = {
        "yaw_mean": float(np.mean(pose_info.yaw[valid_frames])),
        "yaw_std": float(np.std(pose_info.yaw[valid_frames])),
        "pitch_mean": float(np.mean(pose_info.pitch[valid_frames])),
        "pitch_std": float(np.std(pose_info.pitch[valid_frames])),
        "roll_mean": float(np.mean(pose_info.roll[valid_frames])),
        "roll_std": float(np.std(pose_info.roll[valid_frames]))
    }
    
    # Статистика скорректированного сигнала
    signal_stats = {
        "R_mean": float(np.mean(corrected_signals.R[valid_frames])),
        "R_std": float(np.std(corrected_signals.R[valid_frames])),
        "G_mean": float(np.mean(corrected_signals.G[valid_frames])),
        "G_std": float(np.std(corrected_signals.G[valid_frames])),
        "B_mean": float(np.mean(corrected_signals.B[valid_frames])),
        "B_std": float(np.std(corrected_signals.B[valid_frames]))
    }
    
    # Общая статистика
    overall_stats = {
        "valid_frames": int(np.sum(valid_frames)),
        "total_frames": len(valid_frames),
        "valid_ratio": float(pose_info.valid_ratio),
        "mean_roi_weight": float(np.mean(pose_info.roi_weights[valid_frames]))
    }
    
    # Формируем итоговый словарь
    diagnostics = {
        "overall_stats": overall_stats,
        "pose_stats": pose_stats,
        "signal_stats": signal_stats,
        "roi_weights": roi_weights_dict
    }
    
    return diagnostics

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" OPTICAL FLOW + WEBSOCKET METADATA")
    print("="*70 + "\n")
   
    np.random.seed(42)
    num_frames = 100
   
    rgb_signals = {}
    roi_names = ['glabella', 'medial_forehead', 'left_lateral_forehead',
                 'right_lateral_forehead', 'left_malar', 'right_malar', 'upper_nasal_dorsum']
   
    for roi in roi_names:
        rgb_signals[f'{roi}_R'] = np.random.rand(num_frames) * 50 + 100
        rgb_signals[f'{roi}_G'] = np.random.rand(num_frames) * 50 + 130
        rgb_signals[f'{roi}_B'] = np.random.rand(num_frames) * 50 + 80
   
    landmarks = {}
    for i in range(num_frames):
        lm = np.random.rand(468, 3)
        landmarks[f'frame_{i+1}'] = lm
   
    fps = 25.0
    print("Запуск коррекции...")
    corrected_signals, pose_info, snr_w, kurt_w, surf_w, motion_w = correct_pose(rgb_signals, landmarks, fps)
   
    print(f"\n  Валидные кадры: {pose_info.valid_ratio:.1f}%")
    
    
    print("\n" + "="*70)
    print("ТЕСТ ФУНКЦИИ extract_diagnostics_for_websocket()")
    print("="*70)
    
    # Для теста (в реальности не используется)
    snr_weights = np.ones((num_frames, len(roi_names))) * 0.9
    kurtosis_weights = np.ones((num_frames, len(roi_names))) * 0.8
    surface_weights = np.ones((num_frames, len(roi_names))) * 0.5
    motion_weights = np.ones((num_frames, len(roi_names))) * 0.95
    
    diagnostics = extract_diagnostics_for_websocket(
        pose_info, corrected_signals, 
        snr_weights, kurtosis_weights, surface_weights, motion_weights,
        roi_names
    )
    
    import json
    print("\nСгенерированные метаданные для WebSocket:")
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    
    print("\n" + "="*70)
    print("  ТЕСТ ЗАВЕРШЕН")
    print("="*70 + "\n")