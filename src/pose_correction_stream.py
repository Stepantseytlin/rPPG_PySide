"""Каузальная покадровая коррекция RGB-сигнала по позе и качеству ROI."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional

import numpy as np
from scipy import signal
from scipy.stats import kurtosis


ROI_NAMES = (
    "glabella",
    "medial_forehead",
    "left_lateral_forehead",
    "right_lateral_forehead",
    "left_malar",
    "right_malar",
    "upper_nasal_dorsum",
)

ROI_LANDMARK_INDICES: Mapping[str, tuple[int, ...]] = {
    "glabella": (151, 108, 107, 55, 8, 285, 336, 337),
    "medial_forehead": (10, 109, 108, 151, 337, 338),
    "left_lateral_forehead": (67, 103, 104, 105, 66, 107, 108, 109),
    "right_lateral_forehead": (297, 338, 337, 336, 296, 334, 333, 332),
    "left_malar": (126, 100, 118, 117, 116, 123, 147, 187, 205, 203, 129, 209),
    "right_malar": (355, 429, 358, 423, 425, 411, 376, 352, 345, 346, 347, 329),
    "upper_nasal_dorsum": (8, 193, 122, 196, 197, 419, 351, 417),
}

# Порядок сохранен совместимым с пакетной реализацией. Для оценки позы реально
# используются глаза, кончик/переносица носа и подбородок.
POSE_LANDMARK_INDICES = (70, 63, 105, 107, 33, 263, 336, 296, 1, 6, 197, 0, 17, 152)
GLOBAL_MOTION_INDICES = (33, 263, 1, 6, 152)


@dataclass(frozen=True, slots=True)
class PoseCorrectionConfig:
    """Настройки каузальной потоковой коррекции."""

    enabled: bool = True
    kalman_enabled: bool = True
    pose_weight_enabled: bool = True
    snr_weight_enabled: bool = True
    kurtosis_weight_enabled: bool = True
    surface_weight_enabled: bool = True
    motion_weight_enabled: bool = True

    blend_ratio: float = 0.7
    yaw_exclude_deg: float = 30.0
    snr_threshold: float = 0.293
    spectral_band_hz: tuple[float, float] = (0.667, 4.0)
    spectral_window_seconds: float = 5.0
    min_spectral_window_seconds: float = 2.0
    spectral_update_interval_seconds: float = 0.5
    weight_smoothing_frames: int = 3

    process_noise: float = 0.001
    measurement_noise: float = 0.01
    motion_alpha: float = 15.0
    max_tracking_gap_seconds: float = 0.5

    def __post_init__(self) -> None:
        low, high = self.spectral_band_hz
        numeric_values = {
            "blend_ratio": self.blend_ratio,
            "yaw_exclude_deg": self.yaw_exclude_deg,
            "snr_threshold": self.snr_threshold,
            "spectral_window_seconds": self.spectral_window_seconds,
            "min_spectral_window_seconds": self.min_spectral_window_seconds,
            "spectral_update_interval_seconds": self.spectral_update_interval_seconds,
            "process_noise": self.process_noise,
            "measurement_noise": self.measurement_noise,
            "motion_alpha": self.motion_alpha,
            "max_tracking_gap_seconds": self.max_tracking_gap_seconds,
        }
        if not all(np.isfinite(value) for value in (*numeric_values.values(), low, high)):
            raise ValueError("Все числовые параметры коррекции должны быть конечными")
        if not 0.0 <= self.blend_ratio <= 1.0:
            raise ValueError("blend_ratio должен находиться в диапазоне [0, 1]")
        if self.yaw_exclude_deg <= 0:
            raise ValueError("yaw_exclude_deg должен быть положительным")
        if self.snr_threshold < 0:
            raise ValueError("snr_threshold не может быть отрицательным")
        if not 0.0 < low < high:
            raise ValueError("spectral_band_hz должен содержать 0 < low < high")
        if self.spectral_window_seconds <= 0:
            raise ValueError("spectral_window_seconds должен быть положительным")
        if not 0 < self.min_spectral_window_seconds <= self.spectral_window_seconds:
            raise ValueError(
                "min_spectral_window_seconds должен быть положительным и не больше spectral_window_seconds"
            )
        if self.spectral_update_interval_seconds <= 0:
            raise ValueError("spectral_update_interval_seconds должен быть положительным")
        if (
            isinstance(self.weight_smoothing_frames, bool)
            or not isinstance(self.weight_smoothing_frames, (int, np.integer))
            or self.weight_smoothing_frames < 1
        ):
            raise ValueError("weight_smoothing_frames должен быть не меньше 1")
        if self.process_noise <= 0 or self.measurement_noise <= 0:
            raise ValueError("Шумы фильтра Калмана должны быть положительными")
        if self.motion_alpha < 0:
            raise ValueError("motion_alpha не может быть отрицательным")
        if self.max_tracking_gap_seconds <= 0:
            raise ValueError("max_tracking_gap_seconds должен быть положительным")


@dataclass(frozen=True, slots=True)
class ROIObservation:
    """Данные кадра: ROI имеют форму ``(N, 3)`` и порядок каналов RGB."""

    sequence_id: int
    timestamp_seconds: float
    roi_rgb: np.ndarray
    landmarks_xyz: Optional[np.ndarray] = None
    roi_valid: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class CorrectedSample:
    """
    Результат коррекции одного кадра.

    ``valid`` относится к RGB текущего кадра. При ``False`` поле ``rgb``
    содержит только fallback (последнее значение либо нули) и его не следует
    добавлять в FIFO как новое измерение. Валидность позы хранится отдельно в
    ``pose_valid``.
    """

    sequence_id: int
    timestamp_seconds: float
    rgb: np.ndarray
    valid: bool
    pose_valid: bool
    status: str
    sqi_ready: bool
    yaw: float
    pitch: float
    roll: float
    roi_weights: np.ndarray
    component_weights: Mapping[str, np.ndarray] = field(default_factory=dict)


class _LandmarkKalmanFilter:
    """Независимые 2D constant-velocity фильтры для набора landmark'ов."""

    def __init__(self, count: int, process_noise: float, measurement_noise: float):
        self._count = count
        self._process_noise = float(process_noise)
        self._measurement_noise = float(measurement_noise)
        self._state = np.zeros((count, 4), dtype=np.float64)
        self._covariance = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], count, axis=0) * 0.1
        self._initialized = np.zeros(count, dtype=bool)

    def reset(self) -> None:
        self._state.fill(0.0)
        self._covariance[:] = np.eye(4, dtype=np.float64) * 0.1
        self._initialized.fill(False)

    def update(self, measurements: Optional[np.ndarray], dt: float) -> tuple[np.ndarray, np.ndarray]:
        dt = max(float(dt), np.finfo(np.float64).eps)
        transition = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        process_covariance = np.eye(4, dtype=np.float64) * self._process_noise
        measurement_covariance = np.eye(2, dtype=np.float64) * self._measurement_noise
        identity = np.eye(4, dtype=np.float64)

        if measurements is None:
            measurements = np.full((self._count, 2), np.nan, dtype=np.float64)
        measurements = np.asarray(measurements, dtype=np.float64)
        if measurements.shape != (self._count, 2):
            raise ValueError(f"measurements должен иметь форму ({self._count}, 2)")

        for index in range(self._count):
            measured = measurements[index]
            measurement_is_valid = bool(np.all(np.isfinite(measured)))

            if not self._initialized[index]:
                if measurement_is_valid:
                    self._state[index, :2] = measured
                    self._state[index, 2:] = 0.0
                    self._initialized[index] = True
                continue

            state = transition @ self._state[index]
            covariance = transition @ self._covariance[index] @ transition.T + process_covariance

            if measurement_is_valid:
                innovation = measured - observation @ state
                innovation_covariance = observation @ covariance @ observation.T + measurement_covariance
                projected = covariance @ observation.T
                gain = np.linalg.solve(innovation_covariance, projected.T).T
                state = state + gain @ innovation
                residual_transform = identity - gain @ observation
                covariance = (
                    residual_transform @ covariance @ residual_transform.T
                    + gain @ measurement_covariance @ gain.T
                )
                covariance = 0.5 * (covariance + covariance.T)

            self._state[index] = state
            self._covariance[index] = covariance

        return self._state[:, :2].copy(), self._initialized.copy()


class PoseCorrectionStream:
    """
    Каузальная потоковая версия pose correction.

    Каждый вызов :meth:`update` потребляет один кадр и сразу возвращает один
    RGB-отсчет. Спектральные признаки используют только накопленную историю.
    """

    def __init__(
        self,
        fps: float,
        config: Optional[PoseCorrectionConfig] = None,
        roi_names: tuple[str, ...] = ROI_NAMES,
        roi_landmark_indices: Mapping[str, tuple[int, ...]] = ROI_LANDMARK_INDICES,
    ):
        self.fps = float(fps)
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps должен быть конечным положительным числом")

        self.config = config or PoseCorrectionConfig()
        low, high = self.config.spectral_band_hz
        self._snr_active = self.config.enabled and self.config.snr_weight_enabled
        self._kurtosis_active = self.config.enabled and self.config.kurtosis_weight_enabled
        if self._snr_active and high >= self.fps / 2.0:
            raise ValueError("Верхняя частота spectral_band_hz должна быть ниже частоты Найквиста")

        self.roi_names = tuple(roi_names)
        if not self.roi_names:
            raise ValueError("roi_names не может быть пустым")
        if len(set(self.roi_names)) != len(self.roi_names):
            raise ValueError("roi_names не должен содержать дубликаты")
        if set(self.roi_names) != set(roi_landmark_indices):
            raise ValueError("Для каждого ROI должен быть задан набор landmark'ов")
        self.roi_landmark_indices = {
            name: tuple(int(index) for index in roi_landmark_indices[name]) for name in self.roi_names
        }
        if any(len(indices) < 3 for indices in self.roi_landmark_indices.values()):
            raise ValueError("Для вычисления нормали каждой ROI требуется минимум три landmark'а")
        if any(index < 0 for indices in self.roi_landmark_indices.values() for index in indices):
            raise ValueError("Индексы landmark'ов не могут быть отрицательными")

        self._num_roi = len(self.roi_names)
        self._window_frames = max(1, int(round(self.config.spectral_window_seconds * self.fps)))
        self._min_spectral_frames = max(
            3, int(round(self.config.min_spectral_window_seconds * self.fps))
        )
        if self._min_spectral_frames > self._window_frames:
            raise ValueError("Спектральное окно короче минимального окна прогрева")
        self._spectral_update_frames = max(
            1, int(round(self.config.spectral_update_interval_seconds * self.fps))
        )
        self._green_history = [deque(maxlen=self._window_frames) for _ in range(self._num_roi)]
        self._weight_history: deque[np.ndarray] = deque(maxlen=self.config.weight_smoothing_frames)
        self._kalman = _LandmarkKalmanFilter(
            count=len(POSE_LANDMARK_INDICES),
            process_noise=self.config.process_noise,
            measurement_noise=self.config.measurement_noise,
        )
        self._sos = None
        if self._snr_active:
            self._sos = signal.butter(
                4,
                [low / (self.fps / 2.0), high / (self.fps / 2.0)],
                btype="bandpass",
                output="sos",
            )

        self.reset()

    def reset(self) -> None:
        """Полностью сбросить состояние между источниками или сессиями."""

        for history in self._green_history:
            history.clear()
        self._weight_history.clear()
        self._kalman.reset()
        self._previous_landmarks: Optional[np.ndarray] = None
        self._last_timestamp: Optional[float] = None
        self._last_sequence_id: Optional[int] = None
        self._last_rgb: Optional[np.ndarray] = None
        self._last_snr_weights = np.ones(self._num_roi, dtype=np.float64)
        self._last_kurtosis_weights = np.ones(self._num_roi, dtype=np.float64)
        self._last_result: Optional[CorrectedSample] = None
        self._processed_frames = 0
        self._valid_frames = 0
        self._last_landmark_timestamp: Optional[float] = None
        self._last_valid_roi_timestamp: Optional[float] = None
        self._spectral_initialized = False

    @property
    def last_result(self) -> Optional[CorrectedSample]:
        return self._last_result

    @property
    def window_frames(self) -> int:
        return self._window_frames

    @property
    def min_spectral_frames(self) -> int:
        return self._min_spectral_frames

    def diagnostics(self) -> dict[str, object]:
        """Вернуть компактное JSON-совместимое состояние потока."""

        result = self._last_result
        return {
            "processed_frames": self._processed_frames,
            "valid_frames": self._valid_frames,
            "valid_ratio": self._valid_frames / self._processed_frames if self._processed_frames else 0.0,
            "status": result.status if result is not None else "empty",
            "sqi_ready": result.sqi_ready if result is not None else False,
            "pose_valid": result.pose_valid if result is not None else False,
            "yaw": self._finite_or_none(result.yaw) if result is not None else None,
            "pitch": self._finite_or_none(result.pitch) if result is not None else None,
            "roll": self._finite_or_none(result.roll) if result is not None else None,
            "roi_weights": result.roi_weights.tolist() if result is not None else [],
        }

    def update(self, observation: ROIObservation) -> CorrectedSample:
        """Обработать один кадр и вернуть один скорректированный RGB-отсчет."""

        roi_rgb, roi_valid, landmarks = self._validate_observation(observation)
        self._handle_time_boundary(observation)
        has_valid_roi = bool(np.any(roi_valid))

        if not self.config.enabled:
            weights = self._normalize_weights(roi_valid.astype(np.float64), roi_valid)
            corrected_rgb = self._fallback_or_mean_rgb(roi_rgb, roi_valid)
            component_weights = {
                name: np.ones(self._num_roi, dtype=np.float64)
                for name in ("pose", "snr", "kurtosis", "surface", "motion")
            }
            result = self._make_result(
                observation=observation,
                rgb=corrected_rgb,
                valid=has_valid_roi,
                pose_valid=False,
                status="disabled",
                sqi_ready=False,
                yaw=np.nan,
                pitch=np.nan,
                roll=np.nan,
                weights=weights,
                component_weights=component_weights,
            )
            self._commit_result(result, observation, landmarks=None)
            return result

        dt = self._frame_dt(observation.timestamp_seconds)

        if has_valid_roi:
            self._last_valid_roi_timestamp = float(observation.timestamp_seconds)
        elif (
            self._last_valid_roi_timestamp is None
            or observation.timestamp_seconds - self._last_valid_roi_timestamp
            > self.config.max_tracking_gap_seconds
        ):
            self._clear_signal_state()

        for roi_index in range(self._num_roi):
            value = roi_rgb[roi_index, 1] if roi_valid[roi_index] else np.nan
            self._green_history[roi_index].append(float(value))

        pose_measurements = self._extract_pose_measurements(landmarks)
        pose_measured = bool(
            pose_measurements is not None and np.all(np.isfinite(pose_measurements))
        )
        if pose_measured:
            self._last_landmark_timestamp = float(observation.timestamp_seconds)
        elif (
            self._last_landmark_timestamp is None
            or observation.timestamp_seconds - self._last_landmark_timestamp
            > self.config.max_tracking_gap_seconds
        ):
            self._kalman.reset()

        if self.config.kalman_enabled:
            pose_landmarks, initialized = self._kalman.update(pose_measurements, dt)
        else:
            pose_landmarks = pose_measurements
            initialized = (
                np.all(np.isfinite(pose_measurements), axis=1)
                if pose_measurements is not None
                else np.zeros(len(POSE_LANDMARK_INDICES), dtype=bool)
            )

        pose_available = bool(pose_landmarks is not None and np.all(initialized))
        if pose_available:
            yaw, pitch, roll = self._estimate_pose(pose_landmarks)
            pose_available = bool(np.all(np.isfinite([yaw, pitch, roll])))
        else:
            yaw = pitch = roll = np.nan

        global_displacement = self._global_displacement(landmarks)
        pose_weights = self._pose_weights(yaw, pitch, global_displacement, pose_available)
        if self._snr_active or self._kurtosis_active:
            snr_weights, kurtosis_weights, spectral_ready = self._spectral_weights()
        else:
            snr_weights = np.ones(self._num_roi, dtype=np.float64)
            kurtosis_weights = np.ones(self._num_roi, dtype=np.float64)
            spectral_ready = True
        surface_weights = (
            self._surface_weights(landmarks)
            if self.config.enabled and self.config.surface_weight_enabled
            else np.ones(self._num_roi, dtype=np.float64)
        )
        motion_weights = (
            self._motion_weights(landmarks, global_displacement)
            if self.config.enabled and self.config.motion_weight_enabled
            else np.ones(self._num_roi, dtype=np.float64)
        )

        component_weights = {
            "pose": pose_weights if self.config.pose_weight_enabled else np.ones(self._num_roi),
            "snr": snr_weights if self.config.snr_weight_enabled else np.ones(self._num_roi),
            "kurtosis": (
                kurtosis_weights if self.config.kurtosis_weight_enabled else np.ones(self._num_roi)
            ),
            "surface": (
                surface_weights if self.config.surface_weight_enabled else np.ones(self._num_roi)
            ),
            "motion": motion_weights if self.config.motion_weight_enabled else np.ones(self._num_roi),
        }

        if has_valid_roi:
            combined = np.ones(self._num_roi, dtype=np.float64)
            for weights in component_weights.values():
                combined *= np.asarray(weights, dtype=np.float64)
            combined[~roi_valid] = 0.0
            weights = self._normalize_weights(combined, roi_valid)
            self._weight_history.append(weights)
            weights = np.mean(np.stack(tuple(self._weight_history), axis=0), axis=0)
            weights[~roi_valid] = 0.0
            weights = self._normalize_weights(weights, roi_valid)
        elif self._weight_history:
            weights = self._weight_history[-1].copy()
        else:
            weights = np.ones(self._num_roi, dtype=np.float64) / self._num_roi

        if has_valid_roi:
            safe_rgb = np.where(roi_valid[:, None], roi_rgb, 0.0)
            weighted_rgb = np.sum(weights[:, None] * safe_rgb, axis=0)
            original_rgb = np.mean(roi_rgb[roi_valid], axis=0)
            blend = self.config.blend_ratio
            corrected_rgb = blend * weighted_rgb + (1.0 - blend) * original_rgb
            corrected_rgb = np.asarray(corrected_rgb, dtype=np.float64)
            self._last_rgb = corrected_rgb.copy()
        elif self._last_rgb is not None:
            corrected_rgb = self._last_rgb.copy()
        else:
            corrected_rgb = np.zeros(3, dtype=np.float64)

        if not has_valid_roi:
            status = "no_face"
        elif not pose_measured or not pose_available:
            status = "no_landmarks"
        elif not spectral_ready and (
            self.config.snr_weight_enabled or self.config.kurtosis_weight_enabled
        ):
            status = "warmup"
        else:
            status = "ok"

        result = self._make_result(
            observation=observation,
            rgb=corrected_rgb,
            valid=has_valid_roi,
            pose_valid=pose_measured and pose_available,
            status=status,
            sqi_ready=spectral_ready,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            weights=weights,
            component_weights=component_weights,
        )
        self._commit_result(result, observation, landmarks=landmarks)
        return result

    def _fallback_or_mean_rgb(self, roi_rgb: np.ndarray, roi_valid: np.ndarray) -> np.ndarray:
        if np.any(roi_valid):
            value = np.mean(roi_rgb[roi_valid], axis=0)
            self._last_rgb = np.asarray(value, dtype=np.float64).copy()
            return self._last_rgb.copy()
        if self._last_rgb is not None:
            return self._last_rgb.copy()
        return np.zeros(3, dtype=np.float64)

    def _make_result(
        self,
        *,
        observation: ROIObservation,
        rgb: np.ndarray,
        valid: bool,
        pose_valid: bool,
        status: str,
        sqi_ready: bool,
        yaw: float,
        pitch: float,
        roll: float,
        weights: np.ndarray,
        component_weights: Mapping[str, np.ndarray],
    ) -> CorrectedSample:
        immutable_components = MappingProxyType(
            {
                name: self._readonly_array(value)
                for name, value in component_weights.items()
            }
        )
        return CorrectedSample(
            sequence_id=int(observation.sequence_id),
            timestamp_seconds=float(observation.timestamp_seconds),
            rgb=self._readonly_array(rgb),
            valid=bool(valid),
            pose_valid=bool(pose_valid),
            status=status,
            sqi_ready=bool(sqi_ready),
            yaw=float(yaw),
            pitch=float(pitch),
            roll=float(roll),
            roi_weights=self._readonly_array(weights),
            component_weights=immutable_components,
        )

    def _commit_result(
        self,
        result: CorrectedSample,
        observation: ROIObservation,
        landmarks: Optional[np.ndarray],
    ) -> None:
        self._previous_landmarks = landmarks.copy() if landmarks is not None else None
        self._last_timestamp = float(observation.timestamp_seconds)
        self._last_sequence_id = int(observation.sequence_id)
        self._processed_frames += 1
        if result.valid:
            self._valid_frames += 1
        self._last_result = result

    def _clear_signal_state(self) -> None:
        for history in self._green_history:
            history.clear()
        self._weight_history.clear()
        self._last_snr_weights.fill(1.0)
        self._last_kurtosis_weights.fill(1.0)
        self._spectral_initialized = False

    @staticmethod
    def _readonly_array(values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64).copy()
        result.setflags(write=False)
        return result

    def _validate_observation(
        self, observation: ROIObservation
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if not isinstance(observation.sequence_id, (int, np.integer)):
            raise TypeError("sequence_id должен быть целым числом")
        if not np.isfinite(observation.timestamp_seconds):
            raise ValueError("timestamp_seconds должен быть конечным числом")

        roi_rgb = np.asarray(observation.roi_rgb, dtype=np.float64)
        expected_shape = (self._num_roi, 3)
        if roi_rgb.shape != expected_shape:
            raise ValueError(f"roi_rgb должен иметь форму {expected_shape}, получено {roi_rgb.shape}")

        finite_rgb = np.all(np.isfinite(roi_rgb), axis=1)
        if observation.roi_valid is None:
            roi_valid = finite_rgb
        else:
            roi_valid = np.asarray(observation.roi_valid, dtype=bool)
            if roi_valid.shape != (self._num_roi,):
                raise ValueError(f"roi_valid должен иметь форму ({self._num_roi},)")
            roi_valid = roi_valid & finite_rgb

        landmarks: Optional[np.ndarray]
        if observation.landmarks_xyz is None:
            landmarks = None
        else:
            landmarks = np.asarray(observation.landmarks_xyz, dtype=np.float64)
            if landmarks.ndim != 2 or landmarks.shape[1] < 2:
                raise ValueError("landmarks_xyz должен иметь форму (N, 2+) или быть None")

        return roi_rgb, roi_valid, landmarks

    def _handle_time_boundary(self, observation: ROIObservation) -> None:
        if self._last_sequence_id is not None and observation.sequence_id <= self._last_sequence_id:
            raise ValueError("sequence_id должен строго возрастать")
        if self._last_timestamp is None:
            return
        delta = float(observation.timestamp_seconds) - self._last_timestamp
        if delta <= 0:
            raise ValueError("timestamp_seconds должен строго возрастать")
        if delta > self.config.max_tracking_gap_seconds:
            self._reset_tracking_state()

    def _reset_tracking_state(self) -> None:
        self._clear_signal_state()
        self._kalman.reset()
        self._previous_landmarks = None
        self._last_landmark_timestamp = None
        self._last_valid_roi_timestamp = None

    def _frame_dt(self, timestamp_seconds: float) -> float:
        if self._last_timestamp is None:
            return 1.0 / self.fps
        delta = float(timestamp_seconds) - self._last_timestamp
        if delta <= 0 or delta > self.config.max_tracking_gap_seconds:
            return 1.0 / self.fps
        return float(np.clip(delta, 0.25 / self.fps, 4.0 / self.fps))

    def _extract_pose_measurements(self, landmarks: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if landmarks is None or landmarks.shape[0] <= max(POSE_LANDMARK_INDICES):
            return None
        measurements = landmarks[np.asarray(POSE_LANDMARK_INDICES), :2]
        return np.asarray(measurements, dtype=np.float64)

    @staticmethod
    def _estimate_pose(landmarks_14: np.ndarray) -> tuple[float, float, float]:
        left_eye = landmarks_14[4, :2]
        right_eye = landmarks_14[5, :2]
        left_eye_outer = landmarks_14[6, :2]
        right_eye_outer = landmarks_14[7, :2]
        nose_tip = landmarks_14[8, :2]
        nose_bridge = landmarks_14[9, :2]
        chin = landmarks_14[13, :2]

        eye_center = (left_eye + right_eye) / 2.0
        eye_distance = np.linalg.norm(left_eye - right_eye)
        if eye_distance <= 1e-6:
            return np.nan, np.nan, np.nan

        yaw = np.degrees(np.arctan2(nose_tip[0] - eye_center[0], eye_distance * 0.5 + 0.01))
        face_height = chin[1] - nose_bridge[1]
        if abs(face_height) <= 1e-6:
            return np.nan, np.nan, np.nan
        nose_position = (nose_tip[1] - eye_center[1]) / face_height
        pitch = (nose_position - 0.5) * 100.0

        # Сохраняем формулу исходной пакетной реализации: roll считается
        # по внешней паре точек, а yaw/pitch — по внутренней.
        eye_delta = right_eye_outer - left_eye_outer
        roll = np.degrees(np.arctan2(eye_delta[1], eye_delta[0]))
        return (
            float(np.clip(yaw, -90.0, 90.0)),
            float(np.clip(pitch, -60.0, 60.0)),
            float(np.clip(roll, -60.0, 60.0)),
        )

    def _global_displacement(self, landmarks: Optional[np.ndarray]) -> float:
        previous = self._previous_landmarks
        if landmarks is None or previous is None:
            return 0.0
        if landmarks.shape[0] <= max(GLOBAL_MOTION_INDICES) or previous.shape[0] <= max(
            GLOBAL_MOTION_INDICES
        ):
            return 0.0
        current_points = landmarks[np.asarray(GLOBAL_MOTION_INDICES), :2]
        previous_points = previous[np.asarray(GLOBAL_MOTION_INDICES), :2]
        valid = np.all(np.isfinite(current_points), axis=1) & np.all(
            np.isfinite(previous_points), axis=1
        )
        if not np.any(valid):
            return 0.0
        return float(np.mean(np.linalg.norm(current_points[valid] - previous_points[valid], axis=1)))

    def _pose_weights(
        self, yaw: float, pitch: float, displacement: float, pose_available: bool
    ) -> np.ndarray:
        weights = np.ones(self._num_roi, dtype=np.float64)
        if not pose_available:
            return weights

        displacement_factor = max(0.0, 1.0 - displacement / 0.2)
        yaw_factor = max(0.0, 1.0 - abs(yaw) / 75.0)
        pitch_factor = max(0.0, 1.0 - abs(pitch) / 50.0)
        weights *= displacement_factor * min(yaw_factor, pitch_factor)

        name_to_index = {name: index for index, name in enumerate(self.roi_names)}
        if yaw < -15.0:
            factor = 1.0 + yaw / 45.0
            for name in ("right_lateral_forehead", "right_malar"):
                if name in name_to_index:
                    weights[name_to_index[name]] *= factor
        elif yaw > 15.0:
            factor = 1.0 - yaw / 45.0
            for name in ("left_lateral_forehead", "left_malar"):
                if name in name_to_index:
                    weights[name_to_index[name]] *= factor

        if pitch > 10.0:
            factor = 1.0 - pitch / 70.0
            for name in ("medial_forehead", "left_lateral_forehead", "right_lateral_forehead"):
                if name in name_to_index:
                    weights[name_to_index[name]] *= factor

        weights = np.maximum(weights, 0.05)
        if abs(yaw) > self.config.yaw_exclude_deg:
            hidden_names = (
                ("left_lateral_forehead", "left_malar")
                if yaw > 0
                else ("right_lateral_forehead", "right_malar")
            )
            for name in hidden_names:
                if name in name_to_index:
                    weights[name_to_index[name]] = 0.05
        return weights

    def _spectral_weights(self) -> tuple[np.ndarray, np.ndarray, bool]:
        available = min((len(history) for history in self._green_history), default=0)
        if available < self._min_spectral_frames:
            return self._last_snr_weights.copy(), self._last_kurtosis_weights.copy(), False

        histories = [np.asarray(history, dtype=np.float64) for history in self._green_history]
        processable = np.asarray(
            [
                np.count_nonzero(np.isfinite(values)) >= self._min_spectral_frames
                for values in histories
            ],
            dtype=bool,
        )
        if not np.any(processable):
            self._last_snr_weights.fill(1.0)
            self._last_kurtosis_weights.fill(1.0)
            self._spectral_initialized = False
            return self._last_snr_weights.copy(), self._last_kurtosis_weights.copy(), False

        should_update = (
            not self._spectral_initialized
            or (self._processed_frames + 1) % self._spectral_update_frames == 0
        )
        if not should_update:
            return (
                self._last_snr_weights.copy(),
                self._last_kurtosis_weights.copy(),
                self._spectral_initialized,
            )

        snr_weights = np.ones(self._num_roi, dtype=np.float64)
        kurtosis_weights = np.ones(self._num_roi, dtype=np.float64)
        for roi_index, values in enumerate(histories):
            if not processable[roi_index]:
                continue
            series = self._interpolate_finite(values)
            if series is None:
                continue
            if self._snr_active:
                snr_weights[roi_index] = self._compute_snr_weight(series)
            if self._kurtosis_active:
                kurtosis_weights[roi_index] = self._compute_kurtosis_weight(series)

        self._last_snr_weights = snr_weights
        self._last_kurtosis_weights = kurtosis_weights
        self._spectral_initialized = True
        return snr_weights.copy(), kurtosis_weights.copy(), True

    @staticmethod
    def _interpolate_finite(values: np.ndarray) -> Optional[np.ndarray]:
        finite = np.isfinite(values)
        if np.count_nonzero(finite) < 2:
            return None
        indexes = np.arange(values.size)
        return np.interp(indexes, indexes[finite], values[finite])

    def _compute_snr_weight(self, values: np.ndarray) -> float:
        centered = values - np.mean(values)
        if np.std(centered) < 1e-8:
            return 0.01
        try:
            if self._sos is None:
                return 1.0
            filtered = signal.sosfiltfilt(self._sos, centered)
        except ValueError:
            filtered = centered
        frequencies, power = signal.welch(
            filtered,
            fs=self.fps,
            nperseg=min(self._window_frames, filtered.size),
        )
        low, high = self.config.spectral_band_hz
        in_band = (frequencies >= low) & (frequencies <= high)
        signal_power = float(np.sum(power[in_band]))
        total_power = float(np.sum(power))
        if total_power <= 1e-12:
            return 0.01
        snr = signal_power / max(total_power - signal_power, 1e-12)
        if not np.isfinite(snr) or snr < self.config.snr_threshold:
            return 0.01
        return float(np.clip(snr * 2.0, 0.1, 1.0))

    @staticmethod
    def _compute_kurtosis_weight(values: np.ndarray) -> float:
        if values.size < 10 or np.std(values) < 1e-5:
            return 0.5
        value = float(kurtosis(values, fisher=True))
        if not np.isfinite(value):
            return 0.5
        if -0.5 <= value <= 1.5:
            return 1.0
        distance = -0.5 - value if value < -0.5 else value - 1.5
        return float(np.clip(1.0 - distance / 3.0, 0.1, 1.0))

    def _surface_weights(self, landmarks: Optional[np.ndarray]) -> np.ndarray:
        weights = np.ones(self._num_roi, dtype=np.float64)
        if landmarks is None or landmarks.shape[1] < 3:
            return weights
        for roi_index, roi_name in enumerate(self.roi_names):
            indices = self.roi_landmark_indices[roi_name]
            if landmarks.shape[0] <= max(indices):
                weights[roi_index] = 0.5
                continue
            points = landmarks[np.asarray(indices), :3]
            if not np.all(np.isfinite(points)):
                weights[roi_index] = 0.5
                continue
            centered = points - np.mean(points, axis=0)
            try:
                covariance = centered.T @ centered / max(points.shape[0] - 1, 1)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                if (
                    not np.all(np.isfinite(eigenvalues))
                    or eigenvalues[-1] <= 1e-12
                    or eigenvalues[-2] <= 1e-12
                ):
                    weights[roi_index] = 0.5
                    continue
                normal = eigenvectors[:, 0]
                norm = np.linalg.norm(normal)
                if norm <= 1e-12:
                    weights[roi_index] = 0.5
                    continue
                angle = np.degrees(np.arccos(np.clip(abs(normal[2]) / norm, 0.0, 1.0)))
                if angle <= 75.0:
                    weights[roi_index] = np.clip(1.0 - abs(angle - 50.0) / 50.0, 0.3, 1.0)
                else:
                    weights[roi_index] = np.clip(1.0 - (angle - 75.0) / 15.0, 0.05, 0.3)
            except np.linalg.LinAlgError:
                weights[roi_index] = 0.5
        return weights

    def _motion_weights(
        self, landmarks: Optional[np.ndarray], global_displacement: float
    ) -> np.ndarray:
        weights = np.ones(self._num_roi, dtype=np.float64)
        previous = self._previous_landmarks
        if landmarks is None or previous is None:
            return weights
        for roi_index, roi_name in enumerate(self.roi_names):
            indices = self.roi_landmark_indices[roi_name]
            if landmarks.shape[0] <= max(indices) or previous.shape[0] <= max(indices):
                continue
            current_points = landmarks[np.asarray(indices), :2]
            previous_points = previous[np.asarray(indices), :2]
            valid = np.all(np.isfinite(current_points), axis=1) & np.all(
                np.isfinite(previous_points), axis=1
            )
            if not np.any(valid):
                continue
            local_displacement = float(
                np.mean(np.linalg.norm(current_points[valid] - previous_points[valid], axis=1))
            )
            relative_motion = max(0.0, local_displacement - global_displacement)
            weights[roi_index] = np.clip(
                np.exp(-self.config.motion_alpha * relative_motion), 0.01, 1.0
            )
        return weights

    @staticmethod
    def _normalize_weights(weights: np.ndarray, valid: np.ndarray) -> np.ndarray:
        normalized = np.nan_to_num(
            np.asarray(weights, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        normalized = np.maximum(normalized, 0.0)
        normalized[~valid] = 0.0
        total = float(np.sum(normalized))
        if total > 1e-12:
            return normalized / total
        valid_count = int(np.count_nonzero(valid))
        if valid_count:
            return valid.astype(np.float64) / valid_count
        return np.ones(normalized.shape, dtype=np.float64) / normalized.size

    @staticmethod
    def _finite_or_none(value: float) -> Optional[float]:
        return float(value) if np.isfinite(value) else None


# Название Stage удобно для будущего подключения между ROIExtractor2 и FIFO.
PoseCorrectionStage = PoseCorrectionStream


__all__ = [
    "CorrectedSample",
    "PoseCorrectionConfig",
    "PoseCorrectionStage",
    "PoseCorrectionStream",
    "ROIObservation",
    "ROI_LANDMARK_INDICES",
    "ROI_NAMES",
]
