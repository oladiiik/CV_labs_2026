"""Сегментація: порогування, k-means (лаб 4)."""
import cv2
import numpy as np


def threshold_binary(image: np.ndarray, thresh: int = 127, maxval: int = 255) -> np.ndarray:
    """Просте бінарне порогування."""
    _, out = cv2.threshold(image, thresh, maxval, cv2.THRESH_BINARY)
    return out


def threshold_otsu(image: np.ndarray) -> np.ndarray:
    """Порогування методом Отсу (автовибір порогу)."""
    _, out = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def threshold_adaptive(
    image: np.ndarray,
    block_size: int = 11,
    C: int = 2,
    method: int = None,
) -> np.ndarray:
    """Адаптивне порогування. method: ADAPTIVE_THRESH_GAUSSIAN_C або MEAN_C."""
    if method is None:
        method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    return cv2.adaptiveThreshold(image, 255, method, cv2.THRESH_BINARY, block_size, C)


def kmeans_color(image_bgr: np.ndarray, K: int = 3) -> np.ndarray:
    """Сегментація за кольором методом k-means. Повертає зображення з K кольорів. Приймає BGR або BGRA; зберігає альфа якщо є."""
    h, w = image_bgr.shape[:2]
    has_alpha = image_bgr.ndim == 3 and image_bgr.shape[2] == 4
    img = image_bgr[:, :, :3] if has_alpha else image_bgr
    pixels = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(h, w, 3)
    if has_alpha:
        segmented = np.dstack([segmented, image_bgr[:, :, 3]])
    return segmented
