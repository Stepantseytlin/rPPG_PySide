from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from domain.config import Config
from processing.compIF import compute_segment, config_to_dict


class DirectThreadsManager:
    def __init__(self, num_workers: int = 1, lock: threading.RLock | None = None, on_result=None, on_error=None):
        self.lock = lock or threading.RLock()
        self.on_result = on_result
        self.on_error = on_error
        self.config = None
        self.IsactiveThread = 0
        self.pIf = None
        self.fif = None
        self._generation = 0
        self._pending_segments = deque()
        self._pool = ThreadPoolExecutor(max_workers=max(1, num_workers))

    def make_connections(self, num_wrkr: int = 0, ResSlot=None, FinSlot=None):
        pass

    def to_thread(self, segment: np.ndarray, pifObj=None, fif=None):
        with self.lock:

            self.pIf = pifObj
            self.fif = fif
            self.config = fif.config if fif is not None else self.config
            segment = np.asarray(segment, dtype=np.float32).copy()
            
            last_freq = 1.1 if pifObj is None or len(pifObj.values) == 0 else pifObj.values[-1]
            task = (segment, self._generation, last_freq)
            if self.IsactiveThread   :
                self._pending_segments.append(task)
                return
            if len(self._pending_segments)>0:
                # print(f"num of pending segs after pause: {len(self._pending_segments)}")
                self._finish_current_locked()
            self._submit_locked(*task)

    def invalidate(self):
        with self.lock:
            self._generation += 1
            self._pending_segments.clear()
            self.IsactiveThread = 0
            if self.fif is not None:
                self.fif.sta.threadrunning = 0

    def stop(self):
        self.invalidate()
        self._pool.shutdown(wait=False, cancel_futures=True)
        

    def _submit_locked(self, segment: np.ndarray, generation: int, last_freq):
        self.IsactiveThread = 1
        
        if self.fif is not None:
            self.fif.sta.threadrunning = 1
        config_payload = config_to_dict(self.config)
        future = self._pool.submit(compute_segment, segment, config_payload, last_freq)
        future.add_done_callback(lambda done_future: self._future_done(done_future, generation))

    def _future_done(self, future, generation: int):
        try:
            data = future.result()
        except Exception as exc:
            self._handle_error(generation, exc)
            return
        self._handle_result(generation, data)

    def _handle_result(self, generation: int, data: dict[str, Any]):
        with self.lock:
            if generation != self._generation:
                self._finish_current_locked()
                return
            if (self.fif.sta.Pause and self.fif.sta.stopPlotting):
                 pass
            if self.fif is not None and self.pIf is not None:
                frequency = np.asarray(data["frequency"], dtype=float)
                frequency_value = frequency.item() if frequency.shape == () else frequency
                frame = np.asarray(data["frame"])
                to_plot = np.asarray(data["pw"]) if self.fif.config.plotSource == "PW" else frame[:, 1]

                self.fif.CR.append(to_plot)
                self.pIf.values.append(frequency_value)
                if self.fif.pgm: 
                    self.fif.pgm[0] = (np.asarray(data["pgm"]))  
                else:
                    # print(f"initialized pgm array : {len(self.fif.pgm)}")
                    self.fif.pgm.append(np.asarray(data["pgm"]))
                self.fif.sta.pwready = 1
                self.fif.sta.freqready = 1
                self.fif.sta.wrapperoutputready = 1
                self.fif.sta.threadrunning = 0

            if self.on_result is not None:
                self.on_result(data)
            self._finish_current_locked()

    def _handle_error(self, generation: int, exc: Exception):
        with self.lock:
            if generation == self._generation:
                if self.fif is not None:
                    self.fif.sta.threadrunning = 0
                if self.on_error is not None:
                    self.on_error(str(exc))
            self._finish_current_locked()

    def _finish_current_locked(self):
        self.IsactiveThread = 0
        if self.fif.sta.Pause:
            # print(f"num of pending segs before pause: {len(self._pending_segments)}")
            return
        while self._pending_segments:
            
            segment, generation, last_freq = self._pending_segments.popleft()

            if generation == self._generation:
                # print(f"left pend {len(self._pending_segments)}")
                self._submit_locked(segment, generation, last_freq)
                break


@dataclass
class ProcessingService:
    project_root: Path
    config: Config = field(default_factory=lambda: Config(source_type="txt", instant=0,segLength=1024, straplen=20))
    default_source: Path | None = None
    running: bool = False
    camera_enabled: bool = False
    realtime_enabled: bool = True
    source_name: str = ""
    source_path: str = ""
    roi_status: str = "ожидание"
    status: str = "Ожидание источника"
    started_at: float | None = None
    frame_counter: int = 0
    heart_rate: float | None = None

    stat: Any = field(default=None, init=False, repr=False)
    pif: Any = field(default=None, init=False, repr=False)
    fifo: Any = field(default=None, init=False, repr=False)
    fifoworker: Any = field(default=None, init=False, repr=False)
    handler: Any = field(default=None, init=False, repr=False)

    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _shutdown_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _worker_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _source_finished: bool = field(default=False, init=False, repr=False)
    _last_error: str | None = field(default=None, init=False, repr=False)
    _last_frame: np.ndarray | None = field(default=None, init=False, repr=False)
    _frame_version: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        self.project_root = Path(self.project_root)
        if self.default_source is None:
            self.default_source = self.project_root / "sources" / "txt" / "RGB" / "P1H1Labnet.txt"
        self.default_source = Path(self.default_source)
        if not self.source_path:
            self.source_name = self.default_source.name
            self.source_path = str(self.default_source)
            self.config.SourceType = "txt"
            self.roi_status = "не требуется"
            self.status = "Источник: текстовый файл"
        with self._lock:
            self._build_pipeline_locked(reset_runtime=True)
        self._ensure_worker()

    def config_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "cad": self.config.cad,
                "fMa": self.config.fMa,
                "fMi": self.config.fMi,
                "segLength": self.config.segLength,
                "strapLen": self.config.strapLen,
                "SourceType": self.config.SourceType,
                "instant": self.config.instant,
                "plotSource": self.config.plotSource,
            }

    def state_payload(self) -> dict[str, Any]:
        with self._lock:
            progress = self._refresh_progress_locked()
            return {
                "running": self.running,
                "cameraEnabled": self.camera_enabled,
                "realtimeEnabled": self.realtime_enabled,
                "sourceName": self.source_name,
                "sourcePath": self.source_path,
                "sourceType": self.config.SourceType,
                "plotSource": self.config.plotSource,
                "roiStatus": self.roi_status,
                "status": self.status,
                "frameCounter": self.frame_counter,
                "segLength": self.config.segLength,
                "progress": round(progress, 1),
                "heartRate": self.heart_rate,
                "config": self._config_payload_locked(),
                "chartData": self._chart_payload_locked(),
                "frameAvailable": self._last_frame is not None,
                "frameVersion": self._frame_version,
                "lastError": self._last_error,
            }

    def update_config(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.running = False
            for key in ("cad", "fMa", "fMi", "segLength", "strapLen"):
                if key in payload:
                    setattr(self.config, key, int(payload[key]))
            self.status = "Параметры обновлены"
            self._build_pipeline_locked(reset_runtime=True)
        return self.state_payload()

    def toggle_processing(self) -> dict[str, Any]:
        self._ensure_worker()
        with self._lock:
            if self.running:
                self.running = False
                self.started_at = None
                if self.stat is not None:
                    self.stat.Pause = 1
                    self.stat.stopPlotting = 1
                    
                self.status = "Пауза"
                return self.state_payload()

            if self.config.SourceType == "video" and not self.source_path:
                self.status = "Выберите видео"
                return self.state_payload()

            if self._source_finished:
                self._build_pipeline_locked(reset_runtime=True)

            self.running = True
            self.started_at = time.monotonic()
            self._last_error = None
            if self.stat is not None:
                self.stat.Pause = 0
                self.stat.stopPlotting = 0
            if self.handler is not None:
                self.handler.handlerFinished = 0
            self.status = "Обработка"
        return self.state_payload()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self.running = False
            self.started_at = None
            self.frame_counter = 0
            self.heart_rate = None
            self._last_error = None
            self._build_pipeline_locked(reset_runtime=True)
            self.roi_status = "не требуется" if self.config.SourceType == "txt" else "ожидание ROI"
            self.status = "Данные сброшены"
        return self.state_payload()

    def control(self, action: str) -> dict[str, Any]:
        if action == "toggle":
            return self.toggle_processing()
        if action == "reset":
            return self.reset()
        return {"error": f"Unknown action: {action}", "state": self.state_payload()}

    def set_source_type(self, source_type: str) -> dict[str, Any]:
        if source_type not in {"txt", "video", "cam"}:
            return {"error": "sourceType must be txt, video, or cam", "state": self.state_payload()}

        with self._lock:
            self.running = False
            self.started_at = None
            self.config.SourceType = source_type
            self.camera_enabled = source_type == "cam"

            if source_type == "txt":
                self.source_name = self.default_source.name
                self.source_path = str(self.default_source)
                self.roi_status = "не требуется"
                self.status = "Источник: текстовый файл"
            elif source_type == "video":
                current_suffix = Path(self.source_path).suffix.lower() if self.source_path else ""
                if current_suffix not in {".mp4", ".avi", ".mov"}:
                    self.source_name = "Видео не выбрано"
                    self.source_path = ""
                self.roi_status = "ожидание ROI"
                self.status = "Источник: видео"
            else:
                self.source_name = "Камера"
                self.source_path = "0"
                self.roi_status = "ожидание ROI"
                self.status = "Камера выбрана"

            self._build_pipeline_locked(reset_runtime=True)
        return self.state_payload()

    def set_source_file(self, path: str | Path) -> dict[str, Any]:
        source = Path(path)
        with self._lock:
            self.running = False
            self.started_at = None
            self.source_name = source.name
            self.source_path = str(source)
            self.camera_enabled = False

            suffix = source.suffix.lower()
            if suffix in {".txt", ".csv"}:
                self.config.SourceType = "txt"
                self.roi_status = "не требуется"
                self.status = "Источник: текстовый файл" if source == self.default_source else f"TXT загружен: {source.name}"
            elif suffix in {".mp4", ".avi", ".mov"}:
                self.config.SourceType = "video"
                self.roi_status = "ожидание ROI"
                self.status = f"Видео загружено: {source.name}"
            else:
                self.status = f"Файл загружен: {source.name}"

            self._build_pipeline_locked(reset_runtime=True)
        return self.state_payload()

    def set_plot_source(self, plot_source: str) -> dict[str, Any]:
        with self._lock:
            self.config.plotSource = "rPPG" if plot_source == "rPPG" else "PW"
            self.status = "График обновлен"
        return self.state_payload()

    def set_realtime(self, enabled: bool) -> dict[str, Any]:
        with self._lock:
            self.realtime_enabled = bool(enabled)
            self.config.instant = 0 if self.realtime_enabled else 1
            self.status = "Режим реального времени" if self.realtime_enabled else "Пакетный расчет TXT"
            self._build_pipeline_locked(reset_runtime=True)
        return self.state_payload()

    def set_camera(self, enabled: bool) -> dict[str, Any]:
        if enabled:
            return self.set_source_type("cam")
        with self._lock:
            if self.config.SourceType == "cam":
                self.camera_enabled = False
                self.config.SourceType = "txt"
                self.source_name = self.default_source.name
                self.source_path = str(self.default_source)
                self.roi_status = "не требуется"
                self.status = "Камера отключена"
                self._build_pipeline_locked(reset_runtime=True)
        return self.state_payload()
    def set_bicamera(self, enabled: bool) -> dict[str, Any]:
            if enabled:
                return self.set_source_type("video")
            with self._lock:
                if self.config.SourceType == "cam":
                    self.camera_enabled = False
                    self.config.SourceType = "txt"
                    self.source_name = self.default_source.name
                    self.source_path = str(self.default_source)
                    self.roi_status = "не требуется"
                    self.status = "Камера отключена"
                    self._build_pipeline_locked(reset_runtime=True)
            return self.state_payload()
    def shutdown(self):
        self._shutdown_event.set()
        with self._lock:
            self.running = False
            self.started_at = None
            self._stop_pipeline_locked()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    def _ensure_worker(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(target=self._run_loop, name="rppg-web-processing", daemon=True)
        self._worker_thread.start()

    def _run_loop(self):
        while not self._shutdown_event.is_set():
            if not self.running:
                time.sleep(0.05)
                continue
            try:
                delay = self._process_once()
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                    self.status = f"Ошибка обработки: {exc}"
                    self.running = False
                    self.started_at = None
                    if self.stat is not None:
                        self.stat.Pause = 1
                delay = 0.2
            time.sleep(delay*0.001)

    def _process_once(self) -> float:

        with self._lock:
            if self.handler is None:
                self._build_pipeline_locked(reset_runtime=False)
            if self.config.instant == 1 and self.config.SourceType == "txt":
                self.status = "Пакетный расчет TXT"
                handler = self.handler
            else:
                handler = self.handler

        if self.config.instant == 1 and self.config.SourceType == "txt":
            freq, pgm = handler.instance.calcInst()
            with self._lock:
                if freq is not None and pgm is not None:
                    self.fifoworker.InstantSlot(np.asarray(freq), np.asarray(pgm))
                    self._update_heart_rate_locked()
                    self.status = "Пакетный расчет завершен"
                else:
                    self.status = "Нет данных для пакетного расчета"
                self.running = False
                self.started_at = None
                self._source_finished = True
            return 0.05

        with self._lock:
            if self.stat.Pause and self.config.SourceType == "txt":
                return 0.05
            roi, frame = self.handler.instance.nextframe()
            if roi is None and frame is None:
                self.running = False
                self.started_at = None
                self._source_finished = True
                self.status = "Источник завершен"
                if self.stat is not None:
                    if self.config.SourceType != "txt" : self.stat.Pause = 1
                    self.stat.stopPlotting = 0
                return 0.1

            if self.config.SourceType != "txt":
                self._remember_frame_locked(frame)

            if roi is not None:
                self.stat.roiinserted += 1
                self.fifoworker.PSlot(np.asarray(roi))
                self.stat.frameready += 1
                self.roi_status = "найдено" if frame is not None else "поиск ROI"
            elif frame is not None:
                self.fifoworker.PSlot(np.asarray(frame))
                self.roi_status = "не требуется" if self.config.SourceType == "txt" else "поиск ROI"

            self.config.totalFrames += 1
            self.frame_counter = min(self.config.totalFrames, self.config.segLength)
            if self.frame_counter < self.config.segLength:
                self.status = f"Обработка: {self.frame_counter}/{self.config.segLength}"
            else:
                self.status = "Сегмент набран, идет расчет"
            self._update_heart_rate_locked()
            return self._fps_delay_locked()
    def resizeFIFO(self,segLength, strapLen):
        self.stat.Pause = 1
        self.fifo.resize(segLength,strapLen)
        self.config.strapLen = strapLen
        self.config.segLength = segLength
        self.handler.ChangeConfig(self.config)
        self.stat.Pause = 0
    def insertSample(self,RGB):
        with self._lock:
            if self.handler is None:
                self._build_pipeline_locked(reset_runtime=False)
            if self.config.instant == 1 and self.config.SourceType == "txt":
                self.status = "Пакетный расчет TXT"
                handler = self.handler
            else:
                handler = self.handler
        with self._lock:
            self.roi_status = "найдено" if RGB is not None else "поиск ROI"
            self.fifoworker.PSlot(RGB)
        self.config.totalFrames += 1
        self.frame_counter = min(self.config.totalFrames, self.config.segLength)
        if self.frame_counter < self.config.segLength:
            self.status = f"Обработка: {self.frame_counter}/{self.config.segLength}"
        else:
            self.status = "Сегмент набран, идет расчет"
        self._update_heart_rate_locked()
        return self.state_payload()

    def insertframe(self,frame) -> float:
            with self._lock:
                if self.handler is None:
                    self._build_pipeline_locked(reset_runtime=False)
                if self.config.instant == 1 and self.config.SourceType == "txt":
                    self.status = "Пакетный расчет TXT"
                    handler = self.handler
                else:
                    handler = self.handler
    
            # if self.config.instant == 1 and self.config.SourceType == "txt":
            #     freq, pgm = handler.instance.calcInst()
            #     with self._lock:
            #         if freq is not None and pgm is not None:
            #             self.fifoworker.InstantSlot(np.asarray(freq), np.asarray(pgm))
            #             self._update_heart_rate_locked()
            #             self.status = "Пакетный расчет завершен"
            #         else:
            #             self.status = "Нет данных для пакетного расчета"
            #         self.running = False
            #         self.started_at = None
            #         self._source_finished = True
            #     return 0.05
    
            with self._lock:
                roi, frame = self.handler.insertFrame(frame)
                if roi is None and frame is None:
                    self.running = False
                    self.started_at = None
                    self._source_finished = True
                    self.status = "Источник завершен"
                    if self.stat is not None:
                        self.stat.Pause = 1
                        self.stat.stopPlotting = 0
                    return 0.1
    
                if self.config.SourceType != "txt":
                    pass # self._remember_frame_locked(frame)
    
                if roi is not None:
                    self.stat.roiinserted += 1
                    self.fifoworker.PSlot(np.asarray(roi))
                    self.stat.frameready += 1
                    self.roi_status = "найдено" if frame is not None else "поиск ROI"
                elif frame is not None:
                    self.fifoworker.PSlot(np.asarray(frame))
                    self.roi_status = "не требуется" if self.config.SourceType == "txt" else "поиск ROI"
    
                self.config.totalFrames += 1
                self.frame_counter = min(self.config.totalFrames, self.config.segLength)
                if self.frame_counter < self.config.segLength:
                    self.status = f"Обработка: {self.frame_counter}/{self.config.segLength}"
                else:
                    self.status = "Сегмент набран, идет расчет"
                self._update_heart_rate_locked()
                return self.state_payload()
            
    def _build_pipeline_locked(self, reset_runtime: bool):
        self._stop_pipeline_locked()

        from domain.FIFO import FIFO
        from domain.IF import IF
        from domain.stat import Stat
        from src.FIFOWorker import FIFOWorker
        from src.sourceHandler import Handler

        self.stat = Stat()
        self.pif = IF()
        self.config.totalFrames = 0
        self.fifo = FIFO(
            seglength=self.config.segLength,
            strapLen=self.config.strapLen,
            IF=self.pif,
            stat=self.stat,
            config=self.config,
        )
        if hasattr(self.fifo.threadmngr, "stop"):
            self.fifo.threadmngr.stop()

        self.fifo.threadmngr = DirectThreadsManager(
            num_workers=1,
            lock=self._lock,
            on_result=self._on_segment_result_locked,
            on_error=self._on_segment_error_locked,
        )
        
        self.fifoworker = FIFOWorker(self.fifo)
        self.handler = Handler(self.config, None, None, filepath=self._handler_filepath_locked(), stat=self.stat, IS=None)
        self._source_finished = False
        self._last_error = None
        if reset_runtime:
            self.frame_counter = 0
            self.heart_rate = None
            self._last_frame = None
            self._frame_version += 1

    def frame_jpeg(self) -> bytes | None:
        with self._lock:
            if self._last_frame is None:
                return None
            frame = np.asarray(self._last_frame).copy()
        return self._encode_frame_jpeg(frame)

    def _stop_pipeline_locked(self):
        if self.handler is not None:
            self.handler.handlerFinished = 1
        if self.fifo is not None and hasattr(self.fifo, "threadmngr"):
            self.fifo.threadmngr.stop()

    def _remember_frame_locked(self, frame):
        if frame is None:
            return
        arr = np.asarray(frame)
        if arr.ndim < 2:
            return
        if arr.ndim == 3 and arr.shape[2] not in {1, 3, 4}:
            return
        self._last_frame = arr.copy()
        self._frame_version += 1

    def _encode_frame_jpeg(self, frame: np.ndarray) -> bytes | None:
        import cv2 as cv

        if frame.ndim == 2:
            prepared = self._to_uint8_frame(frame)
        elif frame.ndim == 3:
            prepared = self._to_uint8_frame(frame)
            if prepared.shape[2] == 4:
                prepared = prepared[:, :, :3]
            if prepared.shape[2] == 1:
                prepared = prepared[:, :, 0]
            else:
                prepared = cv.cvtColor(prepared, cv.COLOR_RGB2BGR)
        else:
            return None

        ok, buffer = cv.imencode(".jpg", prepared, [int(cv.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None
        return buffer.tobytes()

    def _to_uint8_frame(self, frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.dtype == np.uint8:
            return arr
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _handler_filepath_locked(self):
        if self.config.SourceType == "cam":
            return 0
        return self.source_path or None

    def _fps_delay_locked(self) -> float:
        return 0.001 if self.config.SourceType == "txt" else max(0.001, 1 / max(1, self.config.cad))

    def _on_segment_result_locked(self, data: dict[str, Any]):
        self._update_heart_rate_locked()
        if not self._source_finished:
            self.status = "Расчет сегмента завершен"
        elif len(self.fifo.threadmngr._pending_segments)>0:
            self.status = f"Источник завершен, сегментов в очереди {len(self.fifo.threadmngr._pending_segments)}"

    def _on_segment_error_locked(self, message: str):
        self._last_error = message
        self.status = f"Ошибка расчета: {message}"

    def _refresh_progress_locked(self) -> float:
        if self.config.totalFrames:
            self.frame_counter = min(self.config.totalFrames, self.config.segLength)
        return min(100.0, self.frame_counter / max(1, self.config.segLength) * 100.0)

    def _update_heart_rate_locked(self):
        if self.pif is None or len(self.pif.values) == 0:
            return
        latest = np.asarray(self.pif.values[-1], dtype=float)
        value = float(np.nanmean(latest) * 60.0)
        if np.isfinite(value):
            self.heart_rate = value

    def _config_payload_locked(self) -> dict[str, Any]:
        return {
            "cad": self.config.cad,
            "fMa": self.config.fMa,
            "fMi": self.config.fMi,
            "segLength": self.config.segLength,
            "strapLen": self.config.strapLen,
            "SourceType": self.config.SourceType,
            "instant": self.config.instant,
            "plotSource": self.config.plotSource,
        }

    def _chart_payload_locked(self) -> dict[str, Any]:
        return {
            "wave": self._last_wave_locked(self.config.segLength),
            "heartRate": self._heart_history_locked(),
            "pgm": self._pgm_surface_locked(max_rows=1,max_cols=self.config.segLength, ),
        }

    def _last_wave_locked(self, max_points: int = 600) -> list[float]:
        if self.fifo is None or len(self.fifo.CR) == 0:
            return []
        return self._array_to_series(self.fifo.CR[-1], max_points=max_points)

    def _heart_history_locked(self, max_points: int = 240) -> list[float]:
        if self.pif is None or len(self.pif.values) == 0:
            return []
        values = []
        for value in self.pif.values[-max_points:]:
            arr = np.asarray(value, dtype=float)
            mean_value = float(np.nanmean(arr) * 60.0)
            if np.isfinite(mean_value):
                values.append(round(mean_value, 6))
        return values

    def _pgm_surface_locked(self, max_rows: int = 48, max_cols: int = 120) -> list[list[float]]:
        if self.fifo is None or len(self.fifo.pgm) == 0:
            return []
        rows = []
         
        for item in self.fifo.pgm[-max_rows:]:
            arr = np.asarray(item, dtype=float)
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if arr.size > max_cols:
                indexes = np.linspace(0, arr.size - 1, max_cols).astype(int)
                arr = arr[indexes]
            rows.append([round(float(value), 6) for value in arr])
        return rows

    def _array_to_series(self, values, max_points: int) -> list[float]:
        arr = np.asarray(values, dtype=float).reshape(-1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size > max_points:
            indexes = np.linspace(0, arr.size - 1, max_points).astype(int)
            arr = arr[indexes]
        return [round(float(value), 6) for value in arr]


