"""Введення/виведення: завантаження, збереження, розміри зображень (лаб 1)."""
import cv2
import numpy as np


def load_image(path: str):
    """Завантажити зображення з файлу. Повертає BGR/BGRA numpy array або None.
    PNG з прозорістю завантажуються з альфа-каналом (BGRA) — прозорі пікселі зберігаються."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            with open(path, "rb") as f:
                data = f.read()
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception:
            pass
    return img


def save_image(path: str, image: np.ndarray) -> bool:
    """Зберегти зображення у файл."""
    return cv2.imwrite(path, image)


def get_dimensions(image: np.ndarray) -> tuple:
    """Повернути (height, width) або (height, width, channels)."""
    h, w = image.shape[:2]
    if len(image.shape) >= 3:
        return h, w, image.shape[2]
    return h, w
