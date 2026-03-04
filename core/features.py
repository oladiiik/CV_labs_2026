"""Ключові точки та матчинг (лаб 4)."""
import cv2
import numpy as np


def harris_corners(gray: np.ndarray, block_size: int = 2, ksize: int = 3, k: float = 0.04) -> np.ndarray:
    """Детектор Harris. Повертає карту відгуків (float)."""
    return cv2.cornerHarris(np.float32(gray), block_size, ksize, k)


def orb_detect_compute(image: np.ndarray, mask=None):
    """ORB: детекція ключових точок та обчислення дескрипторів."""
    orb = cv2.ORB_create()
    kp, desc = orb.detectAndCompute(image, mask)
    return kp, desc


def draw_keypoints(image: np.ndarray, keypoints, color=(0, 255, 0)) -> np.ndarray:
    """Візуалізація ключових точок на зображенні. Підтримує BGR та BGRA."""
    if len(image.shape) == 3 and image.shape[2] == 4 and len(color) == 3:
        color = (*color, 255)
    return cv2.drawKeypoints(image, keypoints, None, color=color, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def match_orb(desc1, desc2, k: int = 2):
    """Матчинг ORB-дескрипторів (BFMatcher). k — кількість найкращих для кожного."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=k)
    return matches


def draw_matches(img1, kp1, img2, kp2, matches, match_color=(0, 255, 0)) -> np.ndarray:
    """Намалювати пари відповідностей між двома зображеннями. Підтримує BGR та BGRA."""
    good = []
    for m_list in matches:
        if len(m_list) >= 2:
            m, n = m_list[0], m_list[1]
            if m.distance < 0.75 * n.distance:
                good.append(m)
        elif len(m_list) == 1:
            good.append(m_list[0])
    c = (match_color if len(match_color) == 4 else (*match_color, 255)) if (len(img1.shape) == 3 and img1.shape[2] == 4) else match_color
    single = (255, 0, 0, 255) if (len(img1.shape) == 3 and img1.shape[2] == 4) else (255, 0, 0)
    out = cv2.drawMatches(img1, kp1, img2, kp2, good, None, matchColor=c, singlePointColor=single)
    return out
