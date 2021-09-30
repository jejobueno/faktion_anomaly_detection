import cv2
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['figure.figsize'] = [5, 5]
plt.rcParams.update({'font.size': 18})

A = imread("./assets/data/normal_dice/0/5.jpg", 0)
B = np.mean(A, -1)  # Convert RGB to grayscale
B = imread("./assets/data/normal_dice/0/0.jpg", 0)
C = imread("./assets/data/anomalous_dice/img_17738_cropped.jpg", 0)

print(np.shape(B))
"""
plt.figure()
plt.imshow(256-A) #, cmap='gray_r'
plt.axis('off')
"""
X, y = np.meshgrid(np.arange(1, np.shape(B)[1] + 1), np.arange(1, np.shape(B)[0] + 1))
fig, ax = plt.subplots(nrows=2, ncols=2, subplot_kw=dict(projection='3d'))
ax[0][0].plot_surface(X, y, 256 - B, rstride=1, cstride=1, linewidth=0, alpha=None, antialiased=True, cmap='viridis')
ax[0][0].set_title('surface plot normal dice')

ax[1][1].plot_surface(X, y, 256 - C, rstride=1, cstride=1, linewidth=0, alpha=None, antialiased=True, cmap='viridis')
ax[1][1].set_title('surface plot anormal dice')

ax[0][1].plot_surface(X, y, cv2.absdiff(B, C), rstride=1, cstride=1, linewidth=0, alpha=None, antialiased=True,
                      cmap='viridis')
ax[0][1].set_title('subtraction template dice - bad dice')

ax[1][0].plot_surface(X, y, cv2.absdiff(B, A), rstride=1, cstride=1, linewidth=0, alpha=None, antialiased=True,
                      cmap='viridis')
ax[1][0].set_title('subtraction template dice - good dice')
plt.show()
