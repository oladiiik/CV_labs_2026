"""Колірні моделі та перетворення: BGR↔HSV, grayscale, пікселі, маски (лаб 2)."""
import cv2
import numpy as np


def to_grayscale(image: np.ndarray, keep_alpha: bool = True):
    """Перетворити BGR або BGRA у grayscale."""
    if len(image.shape) == 2:
        return image.copy()
    if image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        if keep_alpha:
            alpha = image[:, :, 3]
            return np.stack([gray, gray, gray, alpha], axis=2)  # BGRA, сірий + альфа
        return gray
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gray_to_bgr(image: np.ndarray):
    """Перетворити grayscale у BGR або залишити BGRA без змін (для візуалізації та малювання)."""
    if len(image.shape) == 3:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def bgr_to_rgb(image: np.ndarray):
    """Перетворити BGR у RGB для відображення. 2D та BGRA не змінюються (для BGRA використовуйте bgra_to_rgba)."""
    if len(image.shape) == 2:
        return image
    if image.shape[2] == 4:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def bgra_to_rgba(image: np.ndarray):
    """Перетворити BGRA у RGBA для відображення з прозорістю в UI/Streamlit."""
    if len(image.shape) != 3 or image.shape[2] != 4:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)


def for_display(image: np.ndarray):
    """Підготувати зображення для Streamlit: BGR→RGB, BGRA→RGBA (прозорість зберігається)."""
    if len(image.shape) == 2:
        return image
    if image.shape[2] == 4:
        return bgra_to_rgba(image)
    return bgr_to_rgb(image)


def get_pixel(image: np.ndarray, x: int, y: int):
    """Отримати значення пікселя (x, y). BGR/BGRA або скаляр для grayscale."""
    if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
        return image[y, x].copy()
    return None


def set_region(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, color):
    """Змінити область зображення на колір. Змінює image in-place. color: скаляр (2D), (B,G,R) або (B,G,R,A). Для BGRA при (B,G,R) альфа = 255."""
    img = image
    if len(img.shape) == 2:
        img[y1:y2, x1:x2] = color if np.isscalar(color) else color[0]
    else:
        c = np.array(color, dtype=img.dtype)
        if c.size != img.shape[2] and img.shape[2] == 4 and c.size == 3:
            c = np.append(c, 255)
        img[y1:y2, x1:x2] = c
    return img


def bgr_to_hsv(image: np.ndarray):
    """Перетворити BGR або BGRA -> HSV (альфа ігнорується)."""
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def hsv_to_bgr(image: np.ndarray):
    """Перетворити HSV -> BGR."""
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


def hsv_for_display(image_bgr: np.ndarray) -> np.ndarray:
    """Перетворити BGR/BGRA -> HSV і повернути зображення для відображення (кольори HSV у BGR/BGRA).
    Для BGRA зберігає альфа — результат можна показувати з прозорістю (RGB/RGBA в інтерфейсі)."""
    hsv = bgr_to_hsv(image_bgr)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        out = np.dstack([out, image_bgr[:, :, 3]])
    return out


def color_mask_hsv(image_bgr: np.ndarray, lower_hsv: tuple, upper_hsv: tuple) -> np.ndarray:
    """Побудувати бінарну маску за діапазоном HSV. lower/upper = (H, S, V). Приймає BGR або BGRA."""
    hsv = bgr_to_hsv(image_bgr)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def bgr_to_hsv_range(bgr_color: tuple, h_tol: int = 15, s_tol: int = 80, v_tol: int = 80):
    """Повертає (lower_hsv, upper_hsv) для сегментації за кольором. bgr_color = (B, G, R)."""
    b, g, r = bgr_color[0], bgr_color[1], bgr_color[2]
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = int(hsv[0, 0, 0]), int(hsv[0, 0, 1]), int(hsv[0, 0, 2])
    lower = (max(0, h - h_tol), max(0, s - s_tol), max(0, v - v_tol))
    upper = (min(180, h + h_tol), min(255, s + s_tol), min(255, v + v_tol))
    return lower, upper


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Застосувати маску до зображення (результат лише там, де mask > 0). Працює для 2D, BGR і BGRA."""
    return cv2.bitwise_and(image, image, mask=mask)
