"""Лабораторна 1: Вступ до OpenCV."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.io import load_image, save_image, get_dimensions
from core.color import to_grayscale


def run_task_1(image_path: str):
    """1. Завантажити зображення."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": f"Не вдалося завантажити: {image_path}"}
    return {"images": [("Завантажене зображення", img)], "text": None}


def run_task_2(image_path: str):
    """2. Вивести його розміри."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": f"Не вдалося завантажити: {image_path}"}
    h, w = get_dimensions(img)[:2]
    ch = f", {img.shape[2]} канали" if len(img.shape) == 3 else ""
    return {"images": [("Зображення", img)], "text": f"Розміри: висота={h}, ширина={w}{ch}"}


def run_task_3(image_path: str, output_path: str = None):
    """3. Зберегти копію."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": f"Не вдалося завантажити: {image_path}"}
    out = output_path or image_path.replace(".", "_copy.")
    ok = save_image(out, img)
    return {"images": [("Копія (збережено)" if ok else "Оригінал", img)], "text": f"Збережено: {out}" if ok else "Помилка збереження"}


def run_task_4(image_path: str):
    """4. Перетворити у grayscale."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": f"Не вдалося завантажити: {image_path}"}
    gray = to_grayscale(img)
    return {"images": [("Grayscale", gray)], "text": None}


def run_task_5(image_path: str):
    """5. Відобразити результат (оригінал + grayscale)."""
    img = load_image(image_path)
    if img is None:
        return {"images": [], "text": f"Не вдалося завантажити: {image_path}"}
    gray = to_grayscale(img)
    return {"images": [("Оригінал", img), ("Grayscale", gray)], "text": None}


def get_tasks():
    return [
        ("Зберегти копію", run_task_3),
        ("Перетворити у grayscale", run_task_4),
    ]
