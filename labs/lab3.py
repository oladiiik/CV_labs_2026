"""Лабораторна 3: Фільтрація, краї, морфологія. Усі пункти підтримують BGR/BGRA (4 канали)."""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.io import load_image
from core.color import to_grayscale, gray_to_bgr
from core.filters import blur, gaussian, median
from core.edges import sobel_uint8, laplacian_uint8, canny, find_contours, draw_contours
from core.morphology import erode, dilate, open_morph, close_morph, morph_gradient, denoise_binary
from core.segmentation import threshold_binary

# Описи алгоритмів: як влаштовані та що роблять візуально (для кожного пункту)
TASK_DESCRIPTIONS = [
    ("Середнє по ядру k×k: кожен піксель замінюється середнім значенням у вікні. Однакові ваги для всіх сусідів.",
     "Згладжування, зменшення різкості та шуму; зображення трохи розмивається."),
    ("Гаусівське згладжування: ваги сусідніх пікселів задаються гаусіаною (більша вага центру). sigma керує ступенем розмиття.",
     "М’яке розмиття, краще зберігає контури ніж box blur; природний вигляд."),
    ("Медіанний фільтр: кожен піксель замінюється медіаною значень у вікні. Нелінійний фільтр.",
     "Добре прибирає імпульсний шум (пікселі) і зберігає різкі краї; менше розмиття ніж blur."),
    ("Порівняння трьох фільтрів (blur, Gaussian, median) з однаковим розміром ядра.",
     "Видно різницю: blur — рівномірне розмиття, Gaussian — м’якше, median — краще зберігає краї."),
    ("Один і той самий blur-фільтр з ядрами 3×3, 5×5, 9×9.",
     "Чим більше ядро, тим сильніше розмиття та згладжування деталей."),
    ("Sobel: обчислення похідних яскравості (dx по горизонталі, dy по вертикалі) через згортку з ядрами Собеля. Laplacian: друга похідна (лапласіан), реагує на зміну градієнта.",
     "Sobel X/Y — вертикальні та горизонтальні краї; Laplacian — контури об’єктів, чутливий до шуму."),
    ("Canny: градієнт (Sobel) → нерівний поріг (low, high) → придушення немаксимумів → гістерезис для з’єднання країв.",
     "Чіткі лінії країв, мало розривів; білі контури на чорному тлі."),
    ("Порівняння Sobel (комбінований), Laplacian та Canny на одному зображенні.",
     "Різна чутливість до країв і шуму; Canny дає найчіткіші контури."),
    ("Бінаризація країв (Canny) → cv2.findContours (дерево контурів) → малювання контурів на оригіналі.",
     "Кольорові замкнені контури об’єктів на зображенні; видно межі областей."),
    ("Те саме, що виявлення контурів: краї Canny + контури на зображенні.",
     "Візуалізація знайдених контурів поверх оригіналу."),
    ("Бінарне порогування: пікселі > thresh стають 255, інакше 0. Один глобальний поріг для всього зображення.",
     "Чорно-біле зображення: світлі області білі, темні чорні; відсікає за яскравістю."),
    ("Ерозія: мінімум у вікні — тонкі білі області зменшуються. Дилатація: максимум у вікні — білі області розширюються.",
     "Ерозія — втоншуюня білих фігур; дилатація — їх розширення; корисні для бінарних масок."),
    ("Відкриття = ерозія + дилатація: прибирає дрібні білі об’єкти. Закриття = дилатація + ерозія: заповнює дрібні діри.",
     "Відкриття — менше дрібних білих плям; закриття — менше чорних дірок у білих областях."),
    ("Морфологічний градієнт = дилатація − ерозія: контур білих областей.",
     "Лише межі білих областей; товщина ліній залежить від розміру ядра."),
    ("Відкриття (прибирає дрібні білі) + закриття (заповнює дрібні діри) на бінарному зображенні.",
     "Очищене бінарне зображення: менше шуму, більш цільні фігури."),
]


def _load_gray(path):
    """Повертає 2D grayscale для обробки країв/порогів (без альфи)."""
    img = load_image(path)
    if img is None:
        return None
    if len(img.shape) == 2:
        return img
    return to_grayscale(img, keep_alpha=False)


def _as_display(img_2d: np.ndarray, ref_img: np.ndarray):
    """Якщо ref_img має 4 канали — повертає 4-канальне зображення (img_2d у B,G,R + альфа з ref); інакше BGR."""
    if ref_img.ndim == 3 and ref_img.shape[2] == 4:
        alpha = ref_img[:, :, 3]
        return np.stack([img_2d, img_2d, img_2d, alpha], axis=2)
    return gray_to_bgr(img_2d)


def run_task_1(image_path: str, ksize: int = 5):
    """1. Застосування blur-фільтра."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    out = blur(img, ksize)
    i = 0
    return {"images": [("Оригінал", img), ("Blur", out)], "text": f"Ядро {ksize}x{ksize}",
            "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_2(image_path: str, ksize: int = 5, sigma_x: float = 0):
    """2. Гаусів фільтр."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    out = gaussian(img, ksize, sigma_x)
    i = 1
    return {"images": [("Оригінал", img), ("Gaussian", out)], "text": f"Ядро {ksize}x{ksize}",
            "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_3(image_path: str, ksize: int = 5):
    """3. Медіанна фільтрація."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    out = median(img, ksize)
    i = 2
    return {"images": [("Оригінал", img), ("Median", out)], "text": f"Ядро {ksize}x{ksize}",
            "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_4(image_path: str, ksize: int = 5):
    """4. Порівняння результатів (blur, Gaussian, median)."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    i = 3
    return {"images": [
        ("Оригінал", img),
        ("Blur", blur(img, ksize)),
        ("Gaussian", gaussian(img, ksize)),
        ("Median", median(img, ksize)),
    ], "text": None, "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_5(image_path: str):
    """5. Аналіз впливу розміру ядра (3, 5, 9)."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    i = 4
    return {"images": [
        ("Оригінал", img),
        ("ksize=3", blur(img, 3)),
        ("ksize=5", blur(img, 5)),
        ("ksize=9", blur(img, 9)),
    ], "text": None, "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_6(image_path: str, ksize: int = 3):
    """6. Оператори Sobel та Laplacian."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    gray = img if img.ndim == 2 else to_grayscale(img, keep_alpha=False)
    sobel_x = sobel_uint8(gray, 1, 0, ksize=ksize)
    sobel_y = sobel_uint8(gray, 0, 1, ksize=ksize)
    lap = laplacian_uint8(gray, ksize=ksize)
    i = 5
    return {"images": [
        ("Grayscale", _as_display(gray, img)),
        ("Sobel X", _as_display(sobel_x, img)),
        ("Sobel Y", _as_display(sobel_y, img)),
        ("Laplacian", _as_display(lap, img)),
    ], "text": None, "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_7(image_path: str, low=50, high=150):
    """7. Детектор Canny."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    gray = img if img.ndim == 2 else to_grayscale(img, keep_alpha=False)
    edges = canny(gray, low, high)
    i = 6
    return {"images": [
        ("Grayscale", _as_display(gray, img)),
        ("Canny", _as_display(edges, img)),
    ], "text": None, "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_8(image_path: str, low: int = 50, high: int = 150):
    """8. Порівняння методів (Sobel, Laplacian, Canny)."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    gray = img if img.ndim == 2 else to_grayscale(img, keep_alpha=False)
    sobel_xy = sobel_uint8(gray, 1, 1)
    lap = laplacian_uint8(gray)
    canny_img = canny(gray, low, high)
    i = 7
    return {"images": [
        ("Grayscale", _as_display(gray, img)),
        ("Sobel", _as_display(sobel_xy, img)),
        ("Laplacian", _as_display(lap, img)),
        ("Canny", _as_display(canny_img, img)),
    ], "text": None, "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_9(image_path: str, low: int = 50, high: int = 150):
    """9. Виявлення контурів на зображенні."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    gray = img if img.ndim == 2 else to_grayscale(img, keep_alpha=False)
    edges = canny(gray, low, high)
    # Для BGRA: шукаємо контури тільки в межах непрозорої області, щоб лінії не виходили за зображення
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        edges = edges.copy()
        edges[alpha == 0] = 0
    contours, _ = find_contours(edges)
    drawn = draw_contours(img, contours)
    i = 8
    return {"images": [
        ("Краї (Canny)", _as_display(edges, img)),
        ("Контури", drawn),
    ], "text": f"Знайдено контурів: {len(contours)}", "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_10(image_path: str, low: int = 50, high: int = 150):
    """10. Візуалізація результатів (краї + контури)."""
    r = run_task_9(image_path, low, high)
    i = 9
    r["algorithm"] = TASK_DESCRIPTIONS[i][0]
    r["visual"] = TASK_DESCRIPTIONS[i][1]
    return r


def run_task_11(image_path: str, thresh: int = 127):
    """11. Порогування зображення."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    gray = img if img.ndim == 2 else to_grayscale(img, keep_alpha=False)
    binary = threshold_binary(gray, thresh)
    i = 10
    return {"images": [
        ("Grayscale", _as_display(gray, img)),
        ("Бінарне порогування", _as_display(binary, img)),
    ], "text": f"Поріг={thresh}", "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_12(image_path: str, thresh: int = 127, ksize: int = 3):
    """12. Ерозія та дилатація."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    gray = img if img.ndim == 2 else to_grayscale(img, keep_alpha=False)
    binary = threshold_binary(gray, thresh)
    eroded = erode(binary, ksize)
    dilated = dilate(binary, ksize)
    i = 11
    return {"images": [
        ("Бінарне", _as_display(binary, img)),
        ("Ерозія", _as_display(eroded, img)),
        ("Дилатація", _as_display(dilated, img)),
    ], "text": None, "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_13(image_path: str, thresh: int = 127, ksize: int = 3):
    """13. Відкриття і закриття."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    gray = img if img.ndim == 2 else to_grayscale(img, keep_alpha=False)
    binary = threshold_binary(gray, thresh)
    opened = open_morph(binary, ksize)
    closed = close_morph(binary, ksize)
    i = 12
    return {"images": [
        ("Бінарне", _as_display(binary, img)),
        ("Відкриття", _as_display(opened, img)),
        ("Закриття", _as_display(closed, img)),
    ], "text": None, "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_14(image_path: str, thresh: int = 127, ksize: int = 3):
    """14. Морфологічний градієнт."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    gray = img if img.ndim == 2 else to_grayscale(img, keep_alpha=False)
    binary = threshold_binary(gray, thresh)
    grad = morph_gradient(binary, ksize)
    i = 13
    return {"images": [
        ("Бінарне", _as_display(binary, img)),
        ("Морфологічний градієнт", _as_display(grad, img)),
    ], "text": None, "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def run_task_15(image_path: str, thresh: int = 127):
    """15. Очищення шуму на бінарному зображенні."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": "Не вдалося завантажити зображення"}
    gray = img if img.ndim == 2 else to_grayscale(img, keep_alpha=False)
    binary = threshold_binary(gray, thresh)
    cleaned = denoise_binary(binary)
    i = 14
    return {"images": [
        ("До очищення", _as_display(binary, img)),
        ("Після очищення", _as_display(cleaned, img)),
    ], "text": None, "algorithm": TASK_DESCRIPTIONS[i][0], "visual": TASK_DESCRIPTIONS[i][1]}


def get_tasks():
    return [
        ("Blur-фільтр", run_task_1),
        ("Гаусів фільтр", run_task_2),
        ("Медіанна фільтрація", run_task_3),
        ("Порівняння результатів", run_task_4),
        ("Вплив розміру ядра", run_task_5),
        ("Sobel та Laplacian", run_task_6),
        ("Детектор Canny", run_task_7),
        ("Порівняння методів країв", run_task_8),
        ("Виявлення контурів", run_task_9),
        ("Візуалізація результатів", run_task_10),
        ("Порогування", run_task_11),
        ("Ерозія та дилатація", run_task_12),
        ("Відкриття і закриття", run_task_13),
        ("Морфологічний градієнт", run_task_14),
        ("Очищення шуму (бінарне)", run_task_15),
    ]
