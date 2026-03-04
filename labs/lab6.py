"""Лабораторна 6: Мініпроєкт — Кольорова сегментація (обраний варіант). Підтримка BGR/BGRA (4 канали) як у лаб 2–3."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from core.io import load_image
from core.color import bgr_to_hsv, color_mask_hsv, apply_mask, gray_to_bgr
from core.segmentation import threshold_binary, threshold_otsu
from core.morphology import denoise_binary
from core.edges import find_contours, draw_contours


def _as_display(img_2d, ref_img):
    """Якщо ref_img має 4 канали — повертає 4-канальне (img_2d у B,G,R + альфа з ref); інакше BGR."""
    if ref_img.ndim == 3 and ref_img.shape[2] == 4:
        alpha = ref_img[:, :, 3]
        return np.stack([img_2d, img_2d, img_2d, alpha], axis=2)
    return gray_to_bgr(img_2d)


def run_color_segmentation(
    image_path: str,
    h_low: int = 0,
    h_high: int = 30,
    s_low: int = 50,
    s_high: int = 255,
    v_low: int = 50,
    v_high: int = 255,
    denoise: bool = True,
):
    """
    BGR -> HSV -> маска -> очищення морфологією -> контури -> візуалізація.
    """
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    hsv = bgr_to_hsv(img)
    mask = color_mask_hsv(img, (h_low, s_low, v_low), (h_high, s_high, v_high))
    if img.ndim == 3 and img.shape[2] == 4:
        mask = mask.copy()
        mask[img[:, :, 3] == 0] = 0
    if denoise:
        mask = denoise_binary(mask)
    segmented = apply_mask(img, mask)
    contours, _ = find_contours(mask)
    drawn = draw_contours(img, contours, color=(0, 255, 0), thickness=2)
    return {
        "images": [
            ("Оригінал", img),
            ("Маска (HSV)", _as_display(mask, img)),
            ("Сегментація за кольором", segmented),
            ("Контури виділених областей", drawn),
        ],
        "text": f"Знайдено контурів: {len(contours)}. Діапазон HSV: H=[{h_low},{h_high}], S=[{s_low},{s_high}], V=[{v_low},{v_high}]",
    }


def run_object_count(
    image_path: str,
    use_otsu: bool = True,
    thresh: int = 127,
    min_area: int = 100,
):
    """
    Простий підрахунок об'єктів: порогування - контури - фільтр за площею - кількість.
    """
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    binary = threshold_otsu(gray) if use_otsu else threshold_binary(gray, thresh)
    if img.ndim == 3 and img.shape[2] == 4:
        binary[img[:, :, 3] == 0] = 0
    contours, _ = find_contours(binary)
    contours = [c for c in contours if cv2.contourArea(c) >= max(1, min_area)]
    # Найбільший контур за площею - зазвичай фон; не рахуємо його
    if len(contours) > 1:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = contours[1:]
    drawn = draw_contours(img, contours, color=(0, 255, 0), thickness=2)
    cv2.putText(
        drawn, f"Count: {len(contours)}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
    )
    return {
        "images": [
            ("Оригінал", img),
            ("Бінарне (поріг)", _as_display(binary, img)),
            ("Контури та кількість", drawn),
        ],
        "text": f"Знайдено об'єктів (площа ≥ {min_area} пікселів): {len(contours)}",
    }


def get_tasks():
    return [
        ("Кольорова сегментація (демо)", lambda path, **kw: run_color_segmentation(path, **kw)),
        ("Підрахунок об'єктів", lambda path, **kw: run_object_count(path, **kw)),
    ]
