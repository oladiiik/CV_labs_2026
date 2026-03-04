"""Лабораторна 5: Відеоаналіз і вебкамера. Робота з відеопотоком з камери."""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import tempfile
from core.video import open_camera, read_frame, get_frame_size, get_fps, frame_diff, create_writer
from core.color import to_grayscale


def run_task_1(device: int = 0, duration_sec: float = 10.0, stream_callback=None, record: bool = False):
    """1. Захоплення відео з камери."""
    cap = open_camera(device)
    if not cap.isOpened():
        return {"images": [], "text": "Не вдалося відкрити камеру", "video_path": None}
    w, h = get_frame_size(cap)
    fps = get_fps(cap)
    n_frames = max(1, int(fps * duration_sec))
    out_path = None
    writer = None
    if record:
        out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        writer = create_writer(out_path, w, h, fps)
    captured = 0
    frame_delay = 1.0 / fps if fps > 0 else 0.04
    for _ in range(n_frames):
        ret, frame = read_frame(cap)
        if not ret:
            break
        if writer is not None:
            writer.write(frame)
        captured += 1
        if stream_callback is not None:
            stream_callback(frame)
        time.sleep(frame_delay)
    cap.release()
    if writer is not None:
        writer.release()
    if captured == 0:
        return {"images": [], "text": "Не вдалося захопити жодного кадру", "video_path": None}
    return {
        "images": [],
        "text": f"Відеопотік: захоплено {captured} кадрів, {fps:.0f} FPS, {w}x{h}" + (" (без запису)" if not record else ""),
        "video_path": out_path,
    }


def run_task_2(device: int = 0, duration_sec: float = 10.0, stream_callback=None, blur_ksize: int = 5, blur_sigma: float = 0.0):
    """2. Обробка кожного кадру (grayscale + blur)."""
    if stream_callback is None:
        return {"images": [], "text": "Запустіть живий перегляд (п. 2)."}
    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    cap = open_camera(device)
    if not cap.isOpened():
        return {"images": [], "text": "Не вдалося відкрити камеру"}
    fps = get_fps(cap)
    n_frames = max(1, int(fps * duration_sec))
    frame_delay = 1.0 / fps if fps > 0 else 0.04
    for _ in range(n_frames):
        ret, frame = read_frame(cap)
        if not ret:
            break
        gray = to_grayscale(frame)
        blurred = cv2.GaussianBlur(frame, (ksize, ksize), blur_sigma)
        stream_callback(frame, gray, blurred)
        time.sleep(frame_delay)
    cap.release()
    return {"images": [], "text": f"Живий перегляд обробки кадру: {duration_sec:.0f} с"}


def run_task_3(device: int = 0, duration_sec: float = 10.0, stream_callback=None):
    """3. Виділення руху (різниця кадрів)."""
    if stream_callback is not None:
        cap = open_camera(device)
        if not cap.isOpened():
            return {"images": [], "text": "Не вдалося відкрити камеру. Запустіть і рухайте рукою перед камерою."}
        fps = get_fps(cap)
        n_frames = max(1, int(fps * duration_sec))
        frame_delay = 1.0 / fps if fps > 0 else 0.04
        prev = None
        for _ in range(n_frames):
            ret, frame = read_frame(cap)
            if not ret:
                break
            if prev is not None:
                diff = frame_diff(prev, frame)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                stream_callback(frame, diff, thresh)
            prev = frame.copy()
            time.sleep(frame_delay)
        cap.release()
        return {"images": [], "text": f"Живий перегляд руху: {duration_sec:.0f} с"}
    cap = open_camera(device)
    if not cap.isOpened():
        return {"images": [], "text": "Не вдалося відкрити камеру. Запустіть і рухайте рукою перед камерою."}
    ret1, frame1 = read_frame(cap)
    ret2, frame2 = read_frame(cap)
    cap.release()
    if not ret1 or not ret2:
        return {"images": [], "text": "Не вдалося прочитати кадри"}
    diff = frame_diff(frame1, frame2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return {"images": [("Кадр 1", frame1), ("Кадр 2", frame2), ("Різниця (рух)", diff), ("Порог різниці", thresh)], "text": None}


def run_task_4(device: int = 0, duration_sec: float = 10.0, stream_callback=None, x1=100, y1=100, x2=300, y2=300):
    """4. Робота з ROI."""
    if stream_callback is not None:
        cap = open_camera(device)
        if not cap.isOpened():
            return {"images": [], "text": "Не вдалося відкрити камеру"}
        fps = get_fps(cap)
        n_frames = max(1, int(fps * duration_sec))
        frame_delay = 1.0 / fps if fps > 0 else 0.04
        for _ in range(n_frames):
            ret, frame = read_frame(cap)
            if not ret:
                break
            h_cap, w_cap = frame.shape[:2]
            x1c, x2c = max(0, min(x1, x2, w_cap - 1)), max(0, min(max(x1, x2), w_cap))
            y1c, y2c = max(0, min(y1, y2, h_cap - 1)), max(0, min(max(y1, y2), h_cap))
            roi = frame[y1c:y2c, x1c:x2c].copy()
            frame_roi = frame.copy()
            cv2.rectangle(frame_roi, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
            stream_callback(frame_roi, roi)
            time.sleep(frame_delay)
        cap.release()
        return {"images": [], "text": f"Живий перегляд ROI: {duration_sec:.0f} с"}
    cap = open_camera(device)
    if not cap.isOpened():
        return {"images": [], "text": "Не вдалося відкрити камеру"}
    ret, frame = read_frame(cap)
    cap.release()
    if not ret:
        return {"images": [], "text": "Не вдалося прочитати кадр"}
    roi = frame[y1:y2, x1:x2]
    frame_roi = frame.copy()
    cv2.rectangle(frame_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return {"images": [("Кадр з ROI", frame_roi), ("Вирізаний ROI", roi)], "text": f"ROI [{x1}:{x2}, {y1}:{y2}]"}


def run_task_5(device: int = 0, num_seconds: float = 2.0, stream_callback=None):
    """5. Збереження відео - читання відеопотоку та запис у файл."""
    cap = open_camera(device)
    if not cap.isOpened():
        return {"images": [], "text": "Не вдалося відкрити камеру", "video_path": None}
    w, h = get_frame_size(cap)
    fps = get_fps(cap) or 25.0
    n_frames = max(1, int(fps * num_seconds))
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = create_writer(out_path, w, h, fps)
    frame_delay = 1.0 / fps if fps > 0 else 0.04
    for _ in range(n_frames):
        ret, frame = read_frame(cap)
        if not ret:
            break
        writer.write(frame)
        if stream_callback is not None:
            stream_callback(frame)
        time.sleep(frame_delay)
    cap.release()
    writer.release()
    ret, first_frame = cv2.VideoCapture(out_path).read()
    return {
        "images": [("Перший кадр збереженого відео", first_frame)] if ret else [],
        "text": f"Відеопотік записано: {out_path}, кадрів: {n_frames}",
        "video_path": out_path,
    }


def get_tasks():
    return [
        ("Захоплення відео з камери", run_task_1),
        ("Обробка кожного кадру", run_task_2),
        ("Виділення руху", run_task_3),
        ("Робота з ROI", run_task_4),
        ("Збереження відео", run_task_5),
    ]
