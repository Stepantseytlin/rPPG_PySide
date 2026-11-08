from __future__ import annotations

import json

import numpy as np
import pytest

from src.pose_correction_stream import (
    ROI_NAMES,
    PoseCorrectionConfig,
    PoseCorrectionStream,
    ROIObservation,
)


_DEFAULT_LANDMARKS = object()


def _roi_rgb(offset: float = 0.0) -> np.ndarray:
    base = np.arange(len(ROI_NAMES), dtype=np.float64)[:, None]
    return np.hstack((100.0 + base, 120.0 + base, 80.0 + base)) + offset


def _landmarks(nose_x: float = 0.5) -> np.ndarray:
    landmarks = np.zeros((468, 3), dtype=np.float64)
    landmarks[33, :2] = (0.4, 0.4)
    landmarks[263, :2] = (0.6, 0.4)
    landmarks[1, :2] = (nose_x, 0.7)
    landmarks[6, :2] = (0.5, 0.2)
    landmarks[152, :2] = (0.5, 0.8)
    landmarks[336, :2] = (0.45, 0.35)
    landmarks[296, :2] = (0.55, 0.35)
    return landmarks


def _observation(
    sequence_id: int = 1,
    timestamp_seconds: float = 0.0,
    roi_rgb: np.ndarray | None = None,
    landmarks: np.ndarray | None | object = _DEFAULT_LANDMARKS,
    roi_valid: np.ndarray | None = None,
) -> ROIObservation:
    return ROIObservation(
        sequence_id=sequence_id,
        timestamp_seconds=timestamp_seconds,
        roi_rgb=_roi_rgb() if roi_rgb is None else roi_rgb,
        landmarks_xyz=_landmarks() if landmarks is _DEFAULT_LANDMARKS else landmarks,
        roi_valid=roi_valid,
    )


def _pose_only_config(**overrides) -> PoseCorrectionConfig:
    values = {
        "kalman_enabled": False,
        "snr_weight_enabled": False,
        "kurtosis_weight_enabled": False,
        "surface_weight_enabled": False,
        "motion_weight_enabled": False,
        "blend_ratio": 1.0,
        "weight_smoothing_frames": 1,
    }
    values.update(overrides)
    return PoseCorrectionConfig(**values)


def test_disabled_stream_returns_plain_mean() -> None:
    stream = PoseCorrectionStream(fps=30.0, config=PoseCorrectionConfig(enabled=False))
    observation = _observation()

    result = stream.update(observation)

    np.testing.assert_allclose(result.rgb, np.mean(observation.roi_rgb, axis=0))
    np.testing.assert_allclose(result.roi_weights, np.ones(7) / 7.0)
    assert result.status == "disabled"
    assert result.valid


def test_result_contract_and_weight_invariants() -> None:
    stream = PoseCorrectionStream(fps=30.0, config=_pose_only_config())

    result = stream.update(_observation())

    assert result.rgb.shape == (3,)
    assert result.roi_weights.shape == (7,)
    assert np.all(np.isfinite(result.rgb))
    assert np.all(np.isfinite(result.roi_weights))
    assert np.all(result.roi_weights >= 0.0)
    assert np.sum(result.roi_weights) == pytest.approx(1.0)
    assert result.sequence_id == 1
    assert result.timestamp_seconds == 0.0
    for component in result.component_weights.values():
        assert component.shape == (7,)
        assert np.all(np.isfinite(component))


def test_result_arrays_and_component_mapping_are_read_only() -> None:
    result = PoseCorrectionStream(fps=30.0, config=_pose_only_config()).update(_observation())

    with pytest.raises(ValueError):
        result.rgb[0] = 0.0
    with pytest.raises(ValueError):
        result.roi_weights[0] = 0.0
    with pytest.raises(ValueError):
        result.component_weights["pose"][0] = 0.0
    with pytest.raises(TypeError):
        result.component_weights["extra"] = np.ones(7)


def test_positive_yaw_downweights_left_side() -> None:
    stream = PoseCorrectionStream(fps=30.0, config=_pose_only_config())

    result = stream.update(_observation(landmarks=_landmarks(nose_x=0.62)))
    weights = result.roi_weights

    assert result.yaw > 30.0
    assert weights[2] < weights[3]
    assert weights[4] < weights[5]


def test_yaw_threshold_is_taken_from_config() -> None:
    observation = _observation(landmarks=_landmarks(nose_x=0.56))
    strict = PoseCorrectionStream(
        fps=30.0, config=_pose_only_config(yaw_exclude_deg=10.0)
    ).update(observation)
    lenient = PoseCorrectionStream(
        fps=30.0, config=_pose_only_config(yaw_exclude_deg=80.0)
    ).update(observation)

    assert strict.roi_weights[2] < lenient.roi_weights[2]
    assert strict.roi_weights[4] < lenient.roi_weights[4]


def test_invalid_roi_is_excluded_after_weight_smoothing() -> None:
    stream = PoseCorrectionStream(fps=30.0, config=_pose_only_config(weight_smoothing_frames=3))
    stream.update(_observation())
    rgb = _roi_rgb(offset=1.0)
    rgb[4] = np.nan
    valid = np.ones(7, dtype=bool)
    valid[4] = False

    result = stream.update(
        _observation(sequence_id=2, timestamp_seconds=1.0 / 30.0, roi_rgb=rgb, roi_valid=valid)
    )

    assert result.roi_weights[4] == 0.0
    assert np.all(np.isfinite(result.rgb))
    assert np.sum(result.roi_weights) == pytest.approx(1.0)


def test_no_face_uses_explicit_hold_last_fallback() -> None:
    stream = PoseCorrectionStream(fps=30.0, config=_pose_only_config())
    first = stream.update(_observation())
    invalid_rgb = np.full((7, 3), np.nan)

    missing = stream.update(
        _observation(
            sequence_id=2,
            timestamp_seconds=1.0 / 30.0,
            roi_rgb=invalid_rgb,
            landmarks=None,
            roi_valid=np.zeros(7, dtype=bool),
        )
    )

    assert not missing.valid
    assert missing.status == "no_face"
    np.testing.assert_allclose(missing.rgb, first.rgb)
    assert np.sum(missing.roi_weights) == pytest.approx(1.0)


def test_valid_rgb_without_landmarks_is_reported_explicitly() -> None:
    stream = PoseCorrectionStream(fps=30.0, config=_pose_only_config())

    result = stream.update(_observation(landmarks=None))

    assert result.valid
    assert not result.pose_valid
    assert result.status == "no_landmarks"
    assert np.isnan(result.yaw)


def test_short_landmark_loss_is_not_reported_as_current_pose() -> None:
    config = _pose_only_config(kalman_enabled=True, max_tracking_gap_seconds=0.11)
    stream = PoseCorrectionStream(fps=10.0, config=config)
    measured = stream.update(_observation(sequence_id=1, timestamp_seconds=0.0))

    predicted = stream.update(
        _observation(sequence_id=2, timestamp_seconds=0.1, landmarks=None)
    )
    expired = stream.update(
        _observation(sequence_id=3, timestamp_seconds=0.2, landmarks=None)
    )

    assert measured.pose_valid
    assert predicted.valid and not predicted.pose_valid
    assert predicted.status == "no_landmarks"
    assert np.isfinite(predicted.yaw)
    assert expired.status == "no_landmarks"
    assert np.isnan(expired.yaw)


def test_no_face_does_not_pollute_weight_smoothing() -> None:
    stream = PoseCorrectionStream(
        fps=30.0,
        config=_pose_only_config(weight_smoothing_frames=3),
    )
    turned = _landmarks(nose_x=0.62)
    first = stream.update(_observation(landmarks=turned))

    stream.update(
        _observation(
            sequence_id=2,
            timestamp_seconds=1.0 / 30.0,
            roi_rgb=np.full((7, 3), np.nan),
            landmarks=None,
            roi_valid=np.zeros(7, dtype=bool),
        )
    )
    recovered = stream.update(
        _observation(
            sequence_id=3,
            timestamp_seconds=2.0 / 30.0,
            landmarks=turned,
        )
    )

    np.testing.assert_allclose(recovered.roi_weights, first.roi_weights)


def test_spectral_warmup_is_explicit() -> None:
    config = PoseCorrectionConfig(
        kalman_enabled=False,
        pose_weight_enabled=False,
        surface_weight_enabled=False,
        motion_weight_enabled=False,
        spectral_window_seconds=0.5,
        min_spectral_window_seconds=0.2,
        spectral_update_interval_seconds=0.1,
    )
    stream = PoseCorrectionStream(fps=10.0, config=config)

    first = stream.update(_observation(sequence_id=1, timestamp_seconds=0.0))
    second = stream.update(_observation(sequence_id=2, timestamp_seconds=0.1))
    third = stream.update(_observation(sequence_id=3, timestamp_seconds=0.2))

    assert not first.sqi_ready
    assert not second.sqi_ready
    assert third.sqi_ready
    np.testing.assert_allclose(first.component_weights["snr"], np.ones(7))
    np.testing.assert_allclose(first.component_weights["kurtosis"], np.ones(7))
    assert np.all(np.isfinite(third.roi_weights))


def test_first_ready_frame_computes_sqi_even_off_interval() -> None:
    config = PoseCorrectionConfig(
        kalman_enabled=False,
        pose_weight_enabled=False,
        surface_weight_enabled=False,
        motion_weight_enabled=False,
        spectral_window_seconds=0.5,
        min_spectral_window_seconds=0.3,
        spectral_update_interval_seconds=0.4,
    )
    stream = PoseCorrectionStream(fps=10.0, config=config)

    stream.update(_observation(sequence_id=1, timestamp_seconds=0.0))
    stream.update(_observation(sequence_id=2, timestamp_seconds=0.1))
    ready = stream.update(_observation(sequence_id=3, timestamp_seconds=0.2))

    assert ready.sqi_ready
    np.testing.assert_allclose(ready.component_weights["snr"], np.full(7, 0.01))
    np.testing.assert_allclose(ready.component_weights["kurtosis"], np.full(7, 0.5))


def test_missing_samples_do_not_count_as_spectral_warmup() -> None:
    config = PoseCorrectionConfig(
        kalman_enabled=False,
        pose_weight_enabled=False,
        surface_weight_enabled=False,
        motion_weight_enabled=False,
        spectral_window_seconds=0.5,
        min_spectral_window_seconds=0.3,
        spectral_update_interval_seconds=0.1,
    )
    stream = PoseCorrectionStream(fps=10.0, config=config)
    invalid_rgb = np.full((7, 3), np.nan)

    for index in range(3):
        result = stream.update(
            _observation(
                sequence_id=index + 1,
                timestamp_seconds=index / 10.0,
                roi_rgb=invalid_rgb,
                landmarks=None,
                roi_valid=np.zeros(7, dtype=bool),
            )
        )

    assert not result.sqi_ready
    np.testing.assert_allclose(result.component_weights["snr"], np.ones(7))


def test_reset_matches_a_fresh_stream() -> None:
    config = _pose_only_config()
    stream = PoseCorrectionStream(fps=30.0, config=config)
    stream.update(_observation())
    stream.update(_observation(sequence_id=2, timestamp_seconds=1.0 / 30.0, roi_rgb=_roi_rgb(5.0)))

    stream.reset()
    reset_result = stream.update(_observation())
    fresh_result = PoseCorrectionStream(fps=30.0, config=config).update(_observation())

    np.testing.assert_allclose(reset_result.rgb, fresh_result.rgb)
    np.testing.assert_allclose(reset_result.roi_weights, fresh_result.roi_weights)
    assert stream.diagnostics()["processed_frames"] == 1


def test_instances_do_not_share_state() -> None:
    config = _pose_only_config()
    first_stream = PoseCorrectionStream(fps=30.0, config=config)
    second_stream = PoseCorrectionStream(fps=30.0, config=config)
    first_stream.update(_observation(landmarks=_landmarks(nose_x=0.62)))

    second_result = second_stream.update(_observation())
    fresh_result = PoseCorrectionStream(fps=30.0, config=config).update(_observation())

    np.testing.assert_allclose(second_result.rgb, fresh_result.rgb)
    np.testing.assert_allclose(second_result.roi_weights, fresh_result.roi_weights)


@pytest.mark.parametrize("shape", [(3,), (6, 3), (7, 2), (7, 4)])
def test_invalid_rgb_shape_is_rejected(shape: tuple[int, ...]) -> None:
    stream = PoseCorrectionStream(fps=30.0, config=_pose_only_config())
    bad_rgb = np.zeros(shape, dtype=np.float64)

    with pytest.raises(ValueError, match="roi_rgb"):
        stream.update(_observation(roi_rgb=bad_rgb))

    assert stream.diagnostics()["processed_frames"] == 0


def test_sequence_id_must_increase() -> None:
    stream = PoseCorrectionStream(fps=30.0, config=_pose_only_config())
    stream.update(_observation(sequence_id=1, timestamp_seconds=0.0))

    with pytest.raises(ValueError, match="sequence_id"):
        stream.update(_observation(sequence_id=1, timestamp_seconds=1.0 / 30.0))


def test_timestamp_must_increase_without_mutating_state() -> None:
    stream = PoseCorrectionStream(fps=30.0, config=_pose_only_config())
    stream.update(_observation(sequence_id=1, timestamp_seconds=0.0))

    with pytest.raises(ValueError, match="timestamp_seconds"):
        stream.update(_observation(sequence_id=2, timestamp_seconds=0.0))

    assert stream.diagnostics()["processed_frames"] == 1
    recovered = stream.update(
        _observation(sequence_id=2, timestamp_seconds=1.0 / 30.0)
    )
    assert recovered.sequence_id == 2


def test_diagnostics_are_strict_json_compatible() -> None:
    stream = PoseCorrectionStream(fps=30.0, config=_pose_only_config())
    stream.update(_observation())

    json.dumps(stream.diagnostics(), allow_nan=False)


def test_low_fps_is_allowed_when_snr_is_disabled() -> None:
    config = _pose_only_config()
    stream = PoseCorrectionStream(fps=5.0, config=config)

    assert stream.update(_observation()).rgb.shape == (3,)


def test_disabled_stream_is_a_real_low_cost_bypass_at_low_fps() -> None:
    stream = PoseCorrectionStream(fps=5.0, config=PoseCorrectionConfig(enabled=False))

    result = stream.update(_observation(landmarks=None))

    assert result.status == "disabled"
    assert not result.pose_valid
    assert not result.sqi_ready


def test_degenerate_surface_geometry_gets_neutral_weight() -> None:
    config = _pose_only_config(
        pose_weight_enabled=False,
        surface_weight_enabled=True,
    )
    stream = PoseCorrectionStream(fps=30.0, config=config)

    result = stream.update(_observation(landmarks=np.zeros((468, 3), dtype=np.float64)))

    np.testing.assert_allclose(result.component_weights["surface"], np.full(7, 0.5))


def test_low_fps_is_rejected_when_snr_is_enabled() -> None:
    with pytest.raises(ValueError, match="Найквиста"):
        PoseCorrectionStream(fps=5.0, config=PoseCorrectionConfig())
