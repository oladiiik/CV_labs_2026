"""Лабораторна 2: Колірні моделі та пікселі."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.io import load_image
from core.color import get_pixel, set_region, bgr_to_hsv, color_mask_hsv, apply_mask, bgr_to_hsv_range


def run_task_1(image_path: str, x: int = 100, y: int = 100):
    """1. Отримати значення пікселя."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": f"Не вдалося завантажити: {image_path}"}
    px = get_pixel(img, x, y)
    if px is None:
        text = f"Координати ({x},{y}) за межами зображення"
    else:
        px_tuple = tuple(px)
        if len(px_tuple) == 4:
            label, rgb_px = "RGBA", (int(px_tuple[2]), int(px_tuple[1]), int(px_tuple[0]), int(px_tuple[3]))
        else:
            label, rgb_px = "RGB", (int(px_tuple[2]), int(px_tuple[1]), int(px_tuple[0]))
        text = f"Піксель ({x}, {y}): {label}={rgb_px}"
    return {"images": [("Зображення", img)], "text": text}


def run_task_2(image_path: str, x1=50, y1=50, x2=150, y2=150, color=(0, 255, 0)):
    """2. Змінити область зображення."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": f"Не вдалося завантажити: {image_path}"}
    img = img.copy()
    set_region(img, x1, y1, x2, y2, color)
    label = "BGRA" if (len(img.shape) == 3 and img.shape[2] == 4) else "BGR"
    return {"images": [("Область змінена (прямокутник)", img)], "text": f"Область [{x1}:{x2}, {y1}:{y2}] заповнена кольором {label}={color}"}


def run_task_3(image_path: str):
    """3. Перетворити BGR -> HSV. Показуємо оригінал (BGR/BGRA) та справжні канали HSV без перетворення в BGR."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": f"Не вдалося завантажити: {image_path}"}
    hsv = bgr_to_hsv(img)
    # H у OpenCV в [0, 180], S і V в [0, 255]. Масштабуємо H -> [0,255] для відображення як окремий канал
    h_scaled = (hsv[:, :, 0].astype(np.uint16) * 255 // 180).astype(np.uint8)
    hsv_show = np.stack([h_scaled, hsv[:, :, 1], hsv[:, :, 2]], axis=2)
    if img.ndim == 3 and img.shape[2] == 4:
        hsv_show = np.dstack([hsv_show, img[:, :, 3]])
    has_alpha = img.ndim == 3 and img.shape[2] == 4
    orig_label = "BGR" if not has_alpha else "BGRA"
    hsv_label = "HSV" if not has_alpha else "HSVA"
    return {"images": [(orig_label, img), (hsv_label, hsv_show)], "text": None}


def run_task_4(image_path: str, h_low=0, h_high=30, s_low=100, s_high=255, v_low=100, v_high=255,
              target_color_bgr=None, h_tol=15, s_tol=80, v_tol=80):
    """4. Маска та сегментація за кольором (діапазон HSV). target_color_bgr=(B,G,R) — колір для сегментації."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": f"Не вдалося завантажити: {image_path}"}
    if target_color_bgr is not None:
        (h_low, s_low, v_low), (h_high, s_high, v_high) = bgr_to_hsv_range(target_color_bgr, h_tol, s_tol, v_tol)
    mask = color_mask_hsv(img, (h_low, s_low, v_low), (h_high, s_high, v_high))
    segmented = apply_mask(img, mask)
    has_alpha = img.ndim == 3 and img.shape[2] == 4
    if has_alpha:
        mask_display = np.stack([mask, mask, mask, img[:, :, 3]], axis=2)
    else:
        mask_display = np.stack([mask, mask, mask], axis=2)
    orig_label = "RGB" if not has_alpha else "RGBA"
    return {"images": [(orig_label, img), ("Маска", mask_display), ("Сегментація", segmented)], "text": None}


def get_tasks():
    return [
        ("Отримати значення пікселя", run_task_1),
        ("Змінити область зображення", run_task_2),
        ("Перетворити BGR -> HSV", run_task_3),
        ("Маска та сегментація за кольором", run_task_4),
    ]
