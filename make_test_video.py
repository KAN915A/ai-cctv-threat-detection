"""Build a short test clip from the ultralytics sample image (bus + people).

The scene is mostly static so tracked persons register as loitering, and
they stand next to a bus so the vehicle-lurking rule fires too.
"""

import cv2
import numpy as np
from ultralytics.utils.downloads import safe_download

safe_download("https://ultralytics.com/images/bus.jpg", file="bus.jpg")
img = cv2.imread("bus.jpg")
img = cv2.resize(img, (640, 480))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("test_clip.mp4", fourcc, 10, (640, 480))

for i in range(10 * 30):  # 30 seconds at 10 fps
    frame = img.copy()
    # slight jitter so it looks like a live feed
    dx = int(3 * np.sin(i / 10))
    frame = np.roll(frame, dx, axis=1)
    out.write(frame)

out.release()
print("Wrote test_clip.mp4")
