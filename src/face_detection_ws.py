# ============================================================================
# ПРОГРАММА ДЛЯ ИЗВЛЕЧЕНИЯ RGB СИГНАЛОВ ИЗ ОБЛАСТЕЙ ЛИЦА
# ПУТЬ ПРОЕКТА: D:\il\python\face_detection_project_ws
# ============================================================================

import cv2
import os
import numpy as np
from scipy.io import savemat
import time
import warnings
import asyncio
import websockets
import json

# ============================================================================
# ИМПОРТ MEDIAPIPE (с проверкой)
# ============================================================================
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    print("MediaPipe успешно импортирован")
except ImportError as e:
    print(f"Ошибка импорта MediaPipe: {e}")
    exit()

# ============================================================================
# НАСТРОЙКИ ПРОЕКТА
# ============================================================================
PROJECT_DIR = r"D:\il\python\face_detection_project_ws"
MODEL_PATH = os.path.join(PROJECT_DIR, 'face_landmarker.task')
VIDEO_PATH = r"D:\il\python\face_detection_project_ws\materiali\Baz_dannux\P1M1\P1M1_edited_obr.mts"

# Путь к папке MATLAB для сохранения итогового файла
MATLAB_PROJECT_DIR = r"D:\MATLAB_Projects\rPPG_Analysis"

# ============================================================================
# РЕЖИМЫ РАБОТЫ
# ============================================================================
OUTPUT_MODE = 'both'  # 'mat', 'websocket', 'both'
ROI_MODE = 'improved'  # выбор режима: 'standard' или 'improved'

# ============================================================================
# ИМПОРТ МОДУЛЯ КОРРЕКЦИИ ПОЗЫ (ТОЛЬКО ДЛЯ РЕЖИМА 'improved')
# ============================================================================
if ROI_MODE == 'improved':
    try:
        from pose_correction_ws import correct_pose, extract_diagnostics_for_websocket
        print("Модуль коррекции позы импортирован (режим improved)")
        APPLY_CORRECTION = True
    except ImportError as e:
        print(f"Модуль pose_correction_ws не найден: {e}")
        print("Коррекция позы не будет применена")
        APPLY_CORRECTION = False
else:
    print(f"Режим: {ROI_MODE} — коррекция позы НЕ применяется")
    APPLY_CORRECTION = False

# ============================================================================
# НАСТРОЙКИ ДЛЯ РЕЖИМА PIXEL_ENSEMBLE
# ============================================================================
PIXEL_ROI_NAME = 'glabella'
PIXEL_CHANNEL = 'G'
MAX_ROI_H = 100
MAX_ROI_W = 100
MAX_PIXELS = MAX_ROI_H * MAX_ROI_W

# Динамические имена файлов
if ROI_MODE == 'standard':
    OUTPUT_MAT_PATH = os.path.join(PROJECT_DIR, 'P1M1_rgb_signals_standard.mat')
    LANDMARKS_PATH = os.path.join(PROJECT_DIR, 'P1M1_landmarks_standard.mat')
elif ROI_MODE == 'improved':
    OUTPUT_MAT_PATH = os.path.join(PROJECT_DIR, 'P1M1_rgb_signals_improved.mat')
    LANDMARKS_PATH = os.path.join(PROJECT_DIR, 'P1M1_landmarks_improved.mat')
elif ROI_MODE == 'raw_rgb':
    OUTPUT_MAT_PATH = os.path.join(PROJECT_DIR, 'P1M1_RGB.mat')
    LANDMARKS_PATH = None
elif ROI_MODE == 'pixel_ensemble':
    OUTPUT_MAT_PATH = os.path.join(PROJECT_DIR, f'P1M1_pixel_ensemble_{PIXEL_ROI_NAME}_{PIXEL_CHANNEL}.mat')
    LANDMARKS_PATH = None
else:
    print(f"Ошибка: неверный режим ROI_MODE = '{ROI_MODE}'")
    exit()

VIDEO_FPS = 30.0
SHOW_LANDMARK_POINTS = (ROI_MODE != 'raw_rgb')

# ============================================================================
# ПРОВЕРКА ФАЙЛОВ
# ============================================================================
print("="*70)
print("ПРОВЕРКА ФАЙЛОВ")
print("="*70)
print(f"Папка проекта: {PROJECT_DIR}")
print(f"\nМодель MediaPipe: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    print(f"Модель найдена. Размер: {os.path.getsize(MODEL_PATH) / (1024*1024):.1f} МБ")
else:
    print("МОДЕЛЬ НЕ НАЙДЕНА")
    exit()
print(f"\nВидеофайл: {VIDEO_PATH}")
if os.path.exists(VIDEO_PATH):
    print(f"Видео найдено. Размер: {os.path.getsize(VIDEO_PATH) / (1024*1024):.1f} МБ")
else:
    print("ВИДЕО НЕ НАЙДЕНО")
    exit()
print(f"\nРежим: {ROI_MODE.upper()}")
print(f"Режим вывода: {OUTPUT_MODE.upper()}")
if ROI_MODE == 'pixel_ensemble':
    print(f"ROI: {PIXEL_ROI_NAME}, канал: {PIXEL_CHANNEL}")
    print(f"Макс. размер ROI: {MAX_ROI_H}x{MAX_ROI_W} = {MAX_PIXELS} пикселей")
    print(f"Метод: PADDING с NaN (без интерполяции, максимальная точность)")
    print(f"Выходной файл: {os.path.basename(OUTPUT_MAT_PATH)}")
elif ROI_MODE != 'raw_rgb':
    print(f"RGB сигналы: {os.path.basename(OUTPUT_MAT_PATH)}")
    print(f"Landmarks: {os.path.basename(LANDMARKS_PATH)}")
else:
    print(f"RAW RGB матрица: {os.path.basename(OUTPUT_MAT_PATH)}")

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ MEDIAPIPE
# ============================================================================
print("\n" + "="*70)
print("ИНИЦИАЛИЗАЦИЯ MEDIAPIPE")
print("="*70)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
base_options.delegate = mp.tasks.BaseOptions.Delegate.CPU

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

try:
    detector = vision.FaceLandmarker.create_from_options(options)
    print("MediaPipe Face Landmarker успешно инициализирован")
except Exception as e:
    print(f"Ошибка инициализации: {e}")
    exit()

# ============================================================================
# НАСТРОЙКИ ROI
# ============================================================================
ROI_KEYPOINTS_3 = {
    'glabella': [151, 108, 107, 55, 8, 285, 336, 337],
    'left_malar': [126, 100, 118, 117, 116, 123, 147, 187, 205, 203, 129, 209],
    'right_malar': [355, 429, 358, 423, 425, 411, 376, 352, 345, 346, 347, 329]
}

ROI_KEYPOINTS_7 = {
    'glabella': [151, 108, 107, 55, 8, 285, 336, 337],
    'medial_forehead': [10, 109, 108, 151, 337, 338],
    'left_lateral_forehead': [67, 103, 104, 105, 66, 107, 108, 109],
    'right_lateral_forehead': [297, 338, 337, 336, 296, 334, 333, 332],
    'left_malar': [126, 100, 118, 117, 116, 123, 147, 187, 205, 203, 129, 209],
    'right_malar': [355, 429, 358, 423, 425, 411, 376, 352, 345, 346, 347, 329],
    'upper_nasal_dorsum': [8, 193, 122, 196, 197, 419, 351, 417]
}

if ROI_MODE == 'standard':
    ROI_KEYPOINTS = ROI_KEYPOINTS_3
    print(f"Набор ROI: СТАНДАРТНЫЙ (3 области)")
elif ROI_MODE in ['improved', 'raw_rgb', 'pixel_ensemble']:
    ROI_KEYPOINTS = ROI_KEYPOINTS_7
    print(f"Набор ROI: УЛУЧШЕННЫЙ (7 областей)")
    if ROI_MODE == 'pixel_ensemble':
        print(f"Для ансамбля используется: {PIXEL_ROI_NAME}")
else:
    print(f"Ошибка: неверный режим")
    exit()

ROI_MARGIN_X = 0.01
ROI_MARGIN_Y = 0.01

# ============================================================================
# ФУНКЦИИ
# ============================================================================
def get_roi_bbox(landmarks, img_width, img_height, keypoint_indices):
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
    margin_x = int(width * ROI_MARGIN_X)
    margin_y = int(height * ROI_MARGIN_Y)
    x = max(0, min_x - margin_x)
    y = max(0, min_y - margin_y)
    w = max(1, width + 2 * margin_x)
    h = max(1, height + 2 * margin_y)
    if x + w > img_width:
        w = img_width - x
    if y + h > img_height:
        h = img_height - y
    return (x, y, w, h)

def draw_landmark_points(img_display, landmarks, img_width, img_height, keypoint_indices, roi_name):
    """Отрисовывает жёлтые точки landmark'ов для указанной ROI"""
    if not SHOW_LANDMARK_POINTS:
        return
    for idx in keypoint_indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            point_x = int(lm.x * img_width)
            point_y = int(lm.y * img_height)
            cv2.circle(img_display, (point_x, point_y), 4, (0, 255, 255), -1)

def interpolate_nans_safely(arr):
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

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ ВИДЕО
# ============================================================================
print("\n" + "="*70)
print("ОТКРЫТИЕ ВИДЕО")
print("="*70)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Не удалось открыть видеофайл")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = VIDEO_FPS
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"Видео открыто")
print(f"FPS: {fps:.2f}")
print(f"Всего кадров: {total_frames}")
print(f"Длительность: {duration:.1f} сек")

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ МАССИВОВ
# ============================================================================
if ROI_MODE == 'raw_rgb':
    all_R, all_G, all_B = [], [], []
elif ROI_MODE == 'pixel_ensemble':
    pixel_ensemble_data = []
else:
    roi_r_values = {name: [] for name in ROI_KEYPOINTS.keys()}
    roi_g_values = {name: [] for name in ROI_KEYPOINTS.keys()}
    roi_b_values = {name: [] for name in ROI_KEYPOINTS.keys()}

frame_numbers = []
timestamps = []
frame_count = 0
successful_frames = 0
faces_detected_count = 0
all_landmarks = []

# ============================================================================
# ОСНОВНОЙ ЦИКЛ
# ============================================================================
print("\n" + "="*70)
print("НАЧИНАЕМ ОБРАБОТКУ (нажмите 'Q' для выхода)")
print("="*70)

if ROI_MODE != 'raw_rgb':
    window_name = 'rPPG - MediaPipe Face Mesh'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

try:
    while True:
        success, img = cap.read()
        if not success:
            print("\nКонец видео")
            break
        if img is None or img.size == 0:
            frame_count += 1
            continue
        
        img_display = img.copy()
        img_height, img_width = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Детекция лица
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = detector.detect(mp_image)
        
        face_detected = False
        
        if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
            face_detected = True
            faces_detected_count += 1
            face_landmarks = detection_result.face_landmarks[0]
            
            # Сохранение landmark'ов
            if ROI_MODE in ['standard', 'improved']:
                landmarks_array = []
                for lm in face_landmarks:
                    landmarks_array.append([lm.x, lm.y, lm.z])
                all_landmarks.append(np.array(landmarks_array))
            
            # Усреднение по ROI с ГАРАНТИРОВАННОЙ синхронизацией по кадрам
            for roi_name, keypoint_indices in ROI_KEYPOINTS.items():
                roi_bbox = get_roi_bbox(face_landmarks, img_width, img_height, keypoint_indices)
                
                if roi_bbox is not None:
                    (x, y, w, h) = roi_bbox
                    
                    # Отрисовка ROI и landmark'ов
                    if ROI_MODE != 'raw_rgb':
                        cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img_display, roi_name, (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        draw_landmark_points(img_display, face_landmarks, img_width, img_height, 
                                            keypoint_indices, roi_name)
                    
                    if w > 0 and h > 0:
                        roi_region = img[y:y+h, x:x+w]
                        # BGR порядок в OpenCV: 0=B, 1=G, 2=R
                        roi_r_values[roi_name].append(np.mean(roi_region[:, :, 2]))
                        roi_g_values[roi_name].append(np.mean(roi_region[:, :, 1]))
                        roi_b_values[roi_name].append(np.mean(roi_region[:, :, 0]))
                    else:
                        # ROI найдена, но имеет нулевой размер -> NaN
                        roi_r_values[roi_name].append(np.nan)
                        roi_g_values[roi_name].append(np.nan)
                        roi_b_values[roi_name].append(np.nan)
                else:
                    # ROI не найдена (например, скрыта при повороте головы) -> NaN
                    roi_r_values[roi_name].append(np.nan)
                    roi_g_values[roi_name].append(np.nan)
                    roi_b_values[roi_name].append(np.nan)
            
            successful_frames += 1
        else:
            # Лицо НЕ найдено в кадре вообще -> NaN для ВСЕХ ROI
            for roi_name in ROI_KEYPOINTS.keys():
                roi_r_values[roi_name].append(np.nan)
                roi_g_values[roi_name].append(np.nan)
                roi_b_values[roi_name].append(np.nan)
        
        frame_numbers.append(frame_count)
        timestamps.append(frame_count / fps)
        
        # Отображение информации
        if ROI_MODE != 'raw_rgb':
            status_text = f"Face: {'YES' if face_detected else 'NO'}"
            status_color = (0, 255, 0) if face_detected else (0, 0, 255)
            cv2.putText(img_display, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(img_display, f"Frame: {frame_count}/{total_frames}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if frame_count > 0:
                detect_rate = (faces_detected_count / frame_count) * 100
                cv2.putText(img_display, f"Detect: {detect_rate:.1f}%", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img_display, f"Mode: {ROI_MODE}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow(window_name, img_display)
        
        # === Обработка закрытия окна через крестик ===
        key = cv2.waitKey(10) & 0xFF
        
        if ROI_MODE != 'raw_rgb':
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\nОкно закрыто пользователем")
                break
        
        if key == ord('q') or key == 27:
            print("\nВыход по команде пользователя")
            break
        
        frame_count += 1
        if frame_count % 500 == 0:
            print(f"Обработано {frame_count}/{total_frames} кадров")

except KeyboardInterrupt:
    print("\nПрервано (Ctrl+C)")
except Exception as e:
    print(f"\nОшибка: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    if ROI_MODE != 'raw_rgb':
        cv2.destroyAllWindows()

# ============================================================================
# СОХРАНЕНИЕ ДАННЫХ
# ============================================================================
print("\n" + "="*70)
print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*70)

if ROI_MODE == 'raw_rgb':
    print(f"\nРежим: RAW RGB — формирование матрицы {len(all_R)}x3")
    all_R = np.array(all_R, dtype=np.float64)
    all_G = np.array(all_G, dtype=np.float64)
    all_B = np.array(all_B, dtype=np.float64)
    
    all_R = interpolate_nans_safely(all_R)
    all_G = interpolate_nans_safely(all_G)
    all_B = interpolate_nans_safely(all_B)
    
    RGB_matrix = np.column_stack([all_R, all_G, all_B])
    print(f"Размер матрицы: {RGB_matrix.shape[0]} строк x {RGB_matrix.shape[1]} столбцов")
    
    data = {'RGB': RGB_matrix}
    savemat(OUTPUT_MAT_PATH, data)
    print(f"Файл сохранён: {OUTPUT_MAT_PATH}")

elif ROI_MODE == 'pixel_ensemble':
    print(f"\nРежим: PIXEL ENSEMBLE — обработка {len(pixel_ensemble_data)} кадров")
    print("Режим pixel_ensemble пока не реализован в этой ветке.")

else:
    if successful_frames > 10:
        print(f"\nСобрано {successful_frames} кадров данных")
        
        # 1. Сохраняем сырые данные с NaN (для отладки)
        data = {
            'fps': float(fps),
            'total_frames': int(frame_count),
            'successful_frames': int(successful_frames),
            'faces_detected': int(faces_detected_count),
            'frame_numbers': np.array(frame_numbers, dtype=np.int32),
            'timestamps': np.array(timestamps, dtype=np.float64),
            'roi_mode': ROI_MODE,
            'roi_names': list(ROI_KEYPOINTS.keys())
        }
        for roi_name in ROI_KEYPOINTS.keys():
            data[f'{roi_name}_R'] = np.array(roi_r_values[roi_name], dtype=np.float64)
            data[f'{roi_name}_G'] = np.array(roi_g_values[roi_name], dtype=np.float64)
            data[f'{roi_name}_B'] = np.array(roi_b_values[roi_name], dtype=np.float64)
        savemat(OUTPUT_MAT_PATH, data)
        print(f"RGB сигналы (с NaN) сохранены: {OUTPUT_MAT_PATH}")
        
        if LANDMARKS_PATH and len(all_landmarks) > 0:
            landmarks_data = {f'frame_{i+1}': lm for i, lm in enumerate(all_landmarks)}
            savemat(LANDMARKS_PATH, landmarks_data)
            print(f"Landmarks сохранены: {LANDMARKS_PATH}")
        
        # ========================================================================
        # 2. ФОРМИРОВАНИЕ PG ДЛЯ MATLAB (УМНОЕ УСРЕДНЕНИЕ + ИНТЕРПОЛЯЦИЯ)
        # ========================================================================
        print("\nФормирование усредненного сигнала PG...")
        
        # Собираем в матрицы формы (num_rois, num_frames)
        all_R_arr = np.array([roi_r_values[name] for name in ROI_KEYPOINTS.keys()], dtype=np.float64)
        all_G_arr = np.array([roi_g_values[name] for name in ROI_KEYPOINTS.keys()], dtype=np.float64)
        all_B_arr = np.array([roi_b_values[name] for name in ROI_KEYPOINTS.keys()], dtype=np.float64)
        
        # Подавляем безобидное предупреждение о пустых срезах (когда все 7 ROI скрыты в одном кадре)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            R_avg = np.nanmean(all_R_arr, axis=0)
            G_avg = np.nanmean(all_G_arr, axis=0)
            B_avg = np.nanmean(all_B_arr, axis=0)
        
        # Финальная безопасная интерполяция только для итогового сигнала, 
        # если лицо было потеряно целиком на 1-2 кадра.
        R_avg = interpolate_nans_safely(R_avg)
        G_avg = interpolate_nans_safely(G_avg)
        B_avg = interpolate_nans_safely(B_avg)
        
        PG = np.column_stack([R_avg, G_avg, B_avg])
        
        video_name = os.path.basename(VIDEO_PATH).split('_')[0]
        output_name = f'EUT_{video_name}_RGBLCY_improved.mat'
        output_path = os.path.join(MATLAB_PROJECT_DIR, output_name)
        
        if not os.path.exists(MATLAB_PROJECT_DIR):
            print(f"Предупреждение: Папка MATLAB не найдена. Сохраняем в папку проекта.")
            output_path = os.path.join(PROJECT_DIR, output_name)
        
        savemat(output_path, {'PG': PG, 'fps': float(fps)})
        print(f"Файл PG сохранён для pw_RT.m: {output_path}")
        print(f"Размер PG: {PG.shape[0]} x {PG.shape[1]} (кадры x каналы)")
        print("="*70)
        
        # ========================================================================
        # ПРИМЕНЕНИЕ КОРРЕКЦИИ ПОЗЫ (Python)
        # ========================================================================
        if APPLY_CORRECTION and ROI_MODE == 'improved':
            print("\n" + "="*70)
            print("ПРИМЕНЕНИЕ КОРРЕКЦИИ ПОЗЫ (Python)")
            print("="*70)
            
            # Для correct_pose нам нужны непрерывные сигналы для фильтра Баттерворта.
            # Мы интерполируем каждую ROI индивидуально ПЕРЕД передачей в модуль коррекции.
            rgb_signals_dict = {}
            for roi_name in ROI_KEYPOINTS.keys():
                r_arr = np.array(roi_r_values[roi_name], dtype=np.float64)
                g_arr = np.array(roi_g_values[roi_name], dtype=np.float64)
                b_arr = np.array(roi_b_values[roi_name], dtype=np.float64)
                
                rgb_signals_dict[f'{roi_name}_R'] = interpolate_nans_safely(r_arr)
                rgb_signals_dict[f'{roi_name}_G'] = interpolate_nans_safely(g_arr)
                rgb_signals_dict[f'{roi_name}_B'] = interpolate_nans_safely(b_arr)
            
            landmarks_dict = {f'frame_{i+1}': lm for i, lm in enumerate(all_landmarks)}
            
            # Вызов коррекции (возвращает 6 значений)
            corrected, pose_info, snr_w, kurt_w, surf_w, motion_w = correct_pose(
                rgb_signals=rgb_signals_dict,
                landmarks=landmarks_dict,
                fps=fps
            )
            
            PG_corrected = np.column_stack([corrected.R, corrected.G, corrected.B])
            corrected_output_name = f'EUT_{video_name}_RGBLCY_custom_corrected.mat'
            corrected_output_path = os.path.join(MATLAB_PROJECT_DIR, corrected_output_name)
            
            if not os.path.exists(MATLAB_PROJECT_DIR):
                corrected_output_path = os.path.join(PROJECT_DIR, corrected_output_name)
            
            # Сохранение скорректированного файла в .mat (если режим mat или both)
            if OUTPUT_MODE in ['mat', 'both']:
                savemat(corrected_output_path, {'PG': PG_corrected, 'fps': float(fps)})
                print(f"\nФайл с коррекцией сохранён: {corrected_output_path}")
                print(f"Размер PG: {PG_corrected.shape[0]} x {PG_corrected.shape[1]} (кадры x каналы)")
                print(f"Валидные кадры: {pose_info.valid_ratio:.1f}%")
                print(f"Средний вес ROI: {np.mean(pose_info.roi_weights[pose_info.valid_frames]):.3f}")
                print("="*70)
            
            # ========================================================================
            #  ОТПРАВКА РЕАЛЬНЫХ ДАННЫХ ДЛЯ СЕРВЕРА
            # ========================================================================
            if OUTPUT_MODE in ['websocket', 'both']:
                print("\n" + "="*70)
                print("ПОДГОТОВКА ДАННЫХ ДЛЯ WEBSOCKET")
                print("="*70)
                
                # 1. Собираем метаданные (диагностику)
                diagnostics = extract_diagnostics_for_websocket(
                    pose_info, corrected, snr_w, kurt_w, surf_w, motion_w, list(ROI_KEYPOINTS.keys())
                )
                
                # 2. Функция для очистки данных от NaN/Inf перед JSON сериализацией
                def clean_for_json(obj):
                    if isinstance(obj, float):
                        if np.isnan(obj) or np.isinf(obj):
                            return 0.0  # Заменяем проблемные числа на 0.0
                        return obj
                    elif isinstance(obj, dict):
                        return {k: clean_for_json(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [clean_for_json(v) for v in obj]
                    return obj

                raw_payload = {
                    "type": "pg_signal",
                    "fps": float(fps),
                    "signal_R": corrected.R.tolist(),
                    "signal_G": corrected.G.tolist(),
                    "signal_B": corrected.B.tolist(),
                    "metadata": diagnostics
                }
                
                # Применяем очистку ко всему словарю
                payload = clean_for_json(raw_payload)
                
                print(f"Размер отправляемого сигнала: {len(corrected.R)} кадров x 3 канала")
                print(f"Объем данных: ~{len(corrected.R) * 3 * 8 / 1024:.1f} КБ")
                
                # 3. Отправляем на сервер (через async функцию)
                uri = "ws://localhost:8765"
                print(f"\n[WEBSOCKET] Подключение к серверу {uri}...")
                
                async def send_signal():
                    try:
                        async with websockets.connect(uri) as websocket:
                            json_data = json.dumps(payload, allow_nan=False)
                            await websocket.send(json_data)
                            print("[WEBSOCKET] Массивы R, G, B и метаданные успешно отправлены на сервер!")
                            
                            response = await asyncio.wait_for(websocket.recv(), timeout=20.0)
                            print(f"[WEBSOCKET] Ответ сервера: {response}")
                    except ConnectionRefusedError:
                        print("[WEBSOCKET] Сервер не запущен! Нужно запустить server.py в другом терминале.")
                    except ValueError as ve:
                        print(f"[WEBSOCKET] ОШИБКА JSON (обнаружен NaN/Inf): {ve}")
                    except Exception as e:
                        print(f"[WEBSOCKET] Ошибка отправки: {repr(e)}")
                
                # Запускаем async функцию
                asyncio.run(send_signal())
    else:
        print("\nНедостаточно данных для сохранения")

detector.close()
print("\n" + "="*70)
print("ПРОГРАММА ЗАВЕРШЕНА")
print("="*70)