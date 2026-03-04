# Спільні модулі для лабораторних з комп'ютерного зору
#
# io           — тільки I/O: завантаження, збереження, get_dimensions
# color        — колірні простори (BGR↔HSV, BGR↔gray), пікселі, маски
# filters      — згладжування: blur, Gaussian, median
# edges        — краї (Sobel, Laplacian, Canny), контури (залежить від color для gray_to_bgr)
# morphology   — ерозія, дилатація, відкриття/закриття, градієнт, denoise_binary
# segmentation — порогування (binary, Otsu, adaptive), k-means
# features     — ключові точки (Harris, ORB), матчинг, візуалізація
# video        — камера, кадри, ROI, frame_diff, збереження (використовує color.to_grayscale)
