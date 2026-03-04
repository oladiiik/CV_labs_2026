"""Виділення країв та контури (лаб 3)."""
import cv2
import numpy as np

from core.color import gray_to_bgr


def sobel(image: np.ndarray, dx: int = 1, dy: int = 1, ksize: int = 3) -> np.ndarray:
    """Оператор Собеля. dx, dy — порядки похідних."""
    return cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=ksize)


def sobel_uint8(image: np.ndarray, dx: int = 1, dy: int = 1, ksize: int = 3) -> np.ndarray:
    """Собель, результат у діапазоні 0–255 для відображення."""
    s = sobel(image, dx, dy, ksize)
    return np.uint8(np.absolute(s))


def laplacian(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Оператор Лапласа."""
    return cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)


def laplacian_uint8(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Лаплас у діапазоні 0–255."""
    lap = laplacian(image, ksize)
    return np.uint8(np.absolute(lap))


def canny(image: np.ndarray, low: float = 50, high: float = 150) -> np.ndarray:
    """Детектор Canny."""
    return cv2.Canny(image, low, high)


def find_contours(binary_image: np.ndarray, mode=None, method=None):
    """Виявити контури на бінарному зображенні. Повертає (contours, hierarchy)."""
    if mode is None:
        mode = cv2.RETR_TREE
    if method is None:
        method = cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(binary_image, mode, method)
    return contours, hierarchy


def draw_contours(image: np.ndarray, contours, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Намалювати контури на копії зображення. Підтримує 2D, BGR та BGRA."""
    img = np.array(image, dtype=np.uint8, order="C", copy=True)
    if img.ndim == 2:
        out = gray_to_bgr(img)
        cv2.drawContours(out, contours, -1, (color[0], color[1], color[2]) if len(color) >= 3 else color, thickness)
        return out
    if img.ndim == 3 and img.shape[2] == 4:
        h, w = img.shape[:2]
        bgr = np.empty((h, w, 3), dtype=np.uint8, order="C")
        bgr[:] = img[:, :, :3]
        alpha = img[:, :, 3].copy()
        color_bgr = (color[0], color[1], color[2]) if len(color) >= 3 else color
        cv2.drawContours(bgr, contours, -1, color_bgr, thickness)
        mask = np.zeros((h, w), dtype=np.uint8, order="C")
        cv2.drawContours(mask, contours, -1, 255, thickness)
        alpha[mask > 0] = 255
        bgr[alpha == 0] = 0
        out = np.empty((h, w, 4), dtype=np.uint8, order="C")
        out[:, :, :3] = bgr
        out[:, :, 3] = alpha
        return out
    cv2.drawContours(img, contours, -1, (color[0], color[1], color[2]) if len(color) >= 3 else color, thickness)
    return img
