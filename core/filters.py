"""Фільтрація та згладжування (лаб 3)."""
import cv2
import numpy as np


def blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Blur-фільтр (середнє по ядру ksize x ksize)."""
    return cv2.blur(image, (ksize, ksize))


def gaussian(image: np.ndarray, ksize: int = 5, sigma_x: float = 0) -> np.ndarray:
    """Гаусів фільтр. sigma_x=0 — авто від ksize."""
    return cv2.GaussianBlur(image, (ksize, ksize), sigma_x)


def median(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Медіанна фільтрація."""
    return cv2.medianBlur(image, ksize)
