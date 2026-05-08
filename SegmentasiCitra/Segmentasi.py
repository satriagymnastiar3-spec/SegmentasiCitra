import cv2
import numpy as np
from matplotlib import pyplot as plt

# =========================
# MEMBACA GAMBAR
# =========================
img = cv2.imread('mazda.jpg')

if img is None:
    print("Gambar tidak ditemukan!")
    exit()

# Konversi BGR ke RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# =========================
# 1. SEGMENTASI THRESHOLD
# =========================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold Binary
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# =========================
# 2. SEGMENTASI K-MEANS
# =========================
pixel_values = img.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Kriteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100, 0.2)

k = 3  # jumlah cluster
_, labels, centers = cv2.kmeans(pixel_values,
                                k,
                                None,
                                criteria,
                                10,
                                cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape(img.shape)

# =========================
# 3. SEGMENTASI WATERSHED
# =========================
gray_ws = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh_ws = cv2.threshold(gray_ws,
                             0,
                             255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh_ws,
                           cv2.MORPH_OPEN,
                           kernel,
                           iterations=2)

# Background
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Foreground
dist_transform = cv2.distanceTransform(opening,
                                       cv2.DIST_L2,
                                       5)

_, sure_fg = cv2.threshold(dist_transform,
                           0.7 * dist_transform.max(),
                           255,
                           0)

sure_fg = np.uint8(sure_fg)

# Unknown region
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
_, markers = cv2.connectedComponents(sure_fg)

markers = markers + 1
markers[unknown == 255] = 0

# Watershed
markers = cv2.watershed(img, markers)

watershed_img = img.copy()
watershed_img[markers == -1] = [255, 0, 0]

watershed_img = cv2.cvtColor(watershed_img,
                             cv2.COLOR_BGR2RGB)

# =========================
# MENAMPILKAN HASIL
# =========================
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("Gambar Asli")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(thresh, cmap='gray')
plt.title("Threshold Segmentation")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(segmented_image)
plt.title("K-Means Segmentation")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(watershed_img)
plt.title("Watershed Segmentation")
plt.axis('off')

plt.tight_layout()
plt.show()