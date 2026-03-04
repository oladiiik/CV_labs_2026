"""Відео: захоплення, обробка кадрів, ROI, збереження (лаб 5)."""
import cv2
import numpy as np

# Конвертація BGR→gray для frame_diff — через color для єдиної логіки
from core.color import to_grayscale


def open_camera(device: int = 0):
    """Відкрити камеру. device=0 — перша вебкамера."""
    return cv2.VideoCapture(device)


def read_frame(cap) -> tuple:
    """Прочитати один кадр. Повертає (success, frame)."""
    return cap.read()


def get_frame_size(cap) -> tuple:
    """(width, height) кадру."""
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return w, h


def get_fps(cap) -> float:
    return cap.get(cv2.CAP_PROP_FPS) or 25.0


def extract_roi(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Вирізати ROI з кадру."""
    return frame[y1:y2, x1:x2].copy()


def frame_diff(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Різниця двох кадрів (для виділення руху). Повертає grayscale. BGR/BGRA підтримуються."""
    if prev.shape[:2] != curr.shape[:2]:
        return np.zeros(curr.shape[:2], dtype=np.uint8)
    prev_g = to_grayscale(prev, keep_alpha=False)
    curr_g = to_grayscale(curr, keep_alpha=False)
    return cv2.absdiff(prev_g, curr_g)


def create_writer(path: str, width: int, height: int, fps: float = 25.0, fourcc=None):
    """Створити VideoWriter для збереження відео. Спробує H.264 (avc1) для браузера, інакше mp4v."""
    if fourcc is not None:
        return cv2.VideoWriter(path, fourcc, fps, (width, height))
    for codec in ("avc1", "H264", "X264", "mp4v"):
        try:
            c4 = cv2.VideoWriter_fourcc(*codec)
            wr = cv2.VideoWriter(path, c4, fps, (width, height))
            if wr.isOpened():
                return wr
        except Exception:
            continue
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
