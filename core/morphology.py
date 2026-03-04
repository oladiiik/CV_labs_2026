"""Морфологічна обробка (лаб 3)."""
import cv2
import numpy as np


def _ensure_kernel(ksize: int = 3):
    return np.ones((ksize, ksize), np.uint8)


def erode(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Ерозія."""
    return cv2.erode(image, _ensure_kernel(ksize))


def dilate(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Дилатація."""
    return cv2.dilate(image, _ensure_kernel(ksize))


def open_morph(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Відкриття (ерозія + дилатація)."""
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, _ensure_kernel(ksize))


def close_morph(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Закриття (дилатація + ерозія)."""
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, _ensure_kernel(ksize))


def morph_gradient(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Морфологічний градієнт (дилатація - ерозія)."""
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, _ensure_kernel(ksize))


def denoise_binary(binary: np.ndarray, open_ksize: int = 2, close_ksize: int = 3) -> np.ndarray:
    """Очищення шуму на бінарному зображенні: відкриття + закриття."""
    out = open_morph(binary, open_ksize)
    out = close_morph(out, close_ksize)
    return out
