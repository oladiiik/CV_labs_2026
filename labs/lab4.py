"""Лабораторна 4: Сегментація, ключові точки, матчинг. Усі пункти підтримують BGR/BGRA (4 канали)."""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from core.io import load_image
from core.color import to_grayscale, gray_to_bgr
from core.segmentation import threshold_binary, threshold_otsu, threshold_adaptive, kmeans_color
from core.features import harris_corners, orb_detect_compute, draw_keypoints, match_orb, draw_matches

# Описи алгоритмів: як влаштовані та що роблять візуально (для кожного пункту)
TASK_DESCRIPTIONS = [
    ("Глобальний поріг: пікселі яскравіші за thresh стають 255, решта 0. Один поріг для всього зображення.",
     "Чорно-біле зображення; проста бінаризація за яскравістю."),
    ("Адаптивне порогування: для кожного пікселя поріг обчислюється по локальному вікну (block_size); C — зсув. Краще для нерівного освітлення.",
     "Чорно-біле з урахуванням локального контрасту; текст та об'єкти читабельніші."),
    ("Метод Оцу: автоматично обирає поріг за максимізацією міжкласової дисперсії (бінарна кластеризація за яскравістю).",
     "Автоматична бінаризація без ручного підбору порога."),
    ("K-means у просторі кольорів (BGR): пікселі кластеризуються в K груп, кожна замінюється центроїдом. Зберігає альфу.",
     "Зменшення кольорів до K; схематичне кольорове зображення."),
    ("Порівняння чотирьох методів: просте порогування (127), Оцу, адаптивне, k-means (K=3).",
     "Візуальне порівняння різних способів сегментації."),
    ("Детектор Harris: відгук кута на основі автокореляційної матриці градієнтів. block_size, ksize, k — параметри.",
     "Кути позначені червоним на зображенні."),
    ("ORB: детектор орієнтованих BRIEF-ознак. Ключові точки + дескриптори для матчингу.",
     "Зелені кола з орієнтацією на характерних точках зображення."),
    ("Демо матчингу: дві половини одного зображення — ORB-дескриптори з обох, knnMatch, фільтр за ratio; лінії з'єднують відповідні точки.",
     "Лінії між відповідними ключовими точками на двох зображеннях."),
    ("Матчинг ознак між двома зображеннями (ORB + BFMatcher + ratio test).",
     "Відповідності між двома зображеннями у вигляді зелених ліній."),
]


def _as_display(img_2d: np.ndarray, ref_img: np.ndarray):
    """Якщо ref_img має 4 канали — повертає 4-канальне зображення (img_2d у B,G,R + альфа з ref); інакше BGR."""
    if ref_img.ndim == 3 and ref_img.shape[2] == 4:
        alpha = ref_img[:, :, 3]
        return np.stack([img_2d, img_2d, img_2d, alpha], axis=2)
    return gray_to_bgr(img_2d)


def run_task_1(image_path: str, thresh: int = 127):
    """1. Просте порогування."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    gray = img if img.ndim == 2 else to_grayscale(img, keep_alpha=False)
    binary = threshold_binary(gray, thresh)
    i = 0
    return {"images": [
        ("Grayscale", _as_display(gray, img)),
        ("Просте порогування", _as_display(binary, img)),
    ], "text": f"Поріг={thresh}", "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_2(image_path: str, block_size: int = 11, C: int = 2):
    """2. Адаптивне порогування."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    gray = img if img.ndim == 2 else to_grayscale(img, keep_alpha=False)
    adaptive = threshold_adaptive(gray, block_size=block_size, C=C)
    i = 1
    return {"images": [
        ("Grayscale", _as_display(gray, img)),
        ("Адаптивне порогування", _as_display(adaptive, img)),
    ], "text": None, "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_3(image_path: str):
    """3. Метод Отсу."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    gray = img if img.ndim == 2 else to_grayscale(img, keep_alpha=False)
    otsu = threshold_otsu(gray)
    i = 2
    return {"images": [
        ("Grayscale", _as_display(gray, img)),
        ("Otsu", _as_display(otsu, img)),
    ], "text": None, "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_4(image_path: str, K: int = 3):
    """4. Сегментація методом k-means."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    segmented = kmeans_color(img, K)
    i = 3
    return {"images": [("Оригінал", img), (f"k-means (K={K})", segmented)], "text": None,
            "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_5(image_path: str):
    """5. Порівняння результатів (просте, Otsu, адаптивне, k-means)."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    gray = img if img.ndim == 2 else to_grayscale(img, keep_alpha=False)
    i = 4
    return {"images": [
        ("Оригінал", img),
        ("Просте (127)", _as_display(threshold_binary(gray, 127), img)),
        ("Otsu", _as_display(threshold_otsu(gray), img)),
        ("Адаптивне", _as_display(threshold_adaptive(gray), img)),
        ("k-means K=3", kmeans_color(img, 3)),
    ], "text": None, "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_6(image_path: str, block_size: int = 2, ksize: int = 3, k: float = 0.04):
    """6. Детектор Harris."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    gray = img if img.ndim == 2 else to_grayscale(img, keep_alpha=False)
    harris = harris_corners(gray, block_size=block_size, ksize=ksize, k=k)
    img_markers = np.ascontiguousarray(img.copy())
    if img_markers.ndim == 2:
        img_markers = gray_to_bgr(img_markers)
    thresh_harris = 0.01 * harris.max()
    marker_color = [0, 0, 255, 255] if (img_markers.ndim == 3 and img_markers.shape[2] == 4) else [0, 0, 255]
    img_markers[harris > thresh_harris] = marker_color
    i = 5
    return {"images": [
        ("Оригінал", img),
        ("Кути Harris (червоні)", img_markers),
    ], "text": None, "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_7(image_path: str):
    """7. ORB-дескриптори."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    gray = img if img.ndim == 2 else to_grayscale(img, keep_alpha=False)
    kp, desc = orb_detect_compute(gray)
    out = draw_keypoints(img, kp)
    i = 6
    return {"images": [("Оригінал", img), ("ORB ключові точки", out)], "text": f"Кількість точок: {len(kp)}",
            "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_9(image_path: str):
    """8. Порівняння зображень (дві половини одного зображення для демо)."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    h, w = img.shape[:2]
    img2 = img[:, : w // 2].copy()
    img1 = img[:, w // 2 :].copy()
    kp1, desc1 = orb_detect_compute(to_grayscale(img1, keep_alpha=False))
    kp2, desc2 = orb_detect_compute(to_grayscale(img2, keep_alpha=False))
    if desc1 is None or desc2 is None or len(kp1) < 2 or len(kp2) < 2:
        i = 7
        return {"images": [("Зображення 1", img1), ("Зображення 2", img2)], "text": "Недостатньо ключових точок для порівняння",
                "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}
    matches = match_orb(desc1, desc2)
    drawn = draw_matches(img1, kp1, img2, kp2, matches)
    i = 7
    return {"images": [("Матчинг між двома частинами зображення", drawn)], "text": None,
            "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_10(image_path_1: str, image_path_2: str = None):
    """9. Матчинг ознак. Якщо image_path_2 не задано — використовується image_path_1 (дві половинки)."""
    if not image_path_2:
        return run_task_9(image_path_1)
    img1 = load_image(image_path_1)
    img2 = load_image(image_path_2)
    if img1 is None or img2 is None:
        return {"images": [], "text": "Не вдалося завантажити одне або обидва зображення"}
    g1, g2 = to_grayscale(img1, keep_alpha=False), to_grayscale(img2, keep_alpha=False)
    kp1, desc1 = orb_detect_compute(g1)
    kp2, desc2 = orb_detect_compute(g2)
    if desc1 is None or desc2 is None or len(kp1) < 2 or len(kp2) < 2:
        i = 8
        return {"images": [("Зображення 1", img1), ("Зображення 2", img2)], "text": "Недостатньо ключових точок",
                "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}
    matches = match_orb(desc1, desc2)
    drawn = draw_matches(img1, kp1, img2, kp2, matches)
    i = 8
    return {"images": [("Матчинг ознак", drawn)], "text": None,
            "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def get_tasks():
    return [
        ("Просте порогування", run_task_1),
        ("Адаптивне порогування", run_task_2),
        ("Метод Отсу", run_task_3),
        ("k-means сегментація", run_task_4),
        ("Порівняння результатів", run_task_5),
        ("Детектор Harris", run_task_6),
        ("ORB-дескриптори", run_task_7),
        ("Порівняння зображень", run_task_9),
        ("Матчинг ознак", run_task_10),
    ]
