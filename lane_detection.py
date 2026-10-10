"""Pasul 3: detectia benzilor cu computer vision clasic.

Etapa 1 (acest fisier, deocamdata): filtrarea - din imaginea camerei pastram
doar pixelii care seamana cu un marcaj rutier.

Demo pe imagini salvate (o tasta trece la urmatoarea, ESC iese):
    venv\\Scripts\\python.exe lane_detection.py data\\frames\\frame_0060.png
"""

import sys

import cv2
import numpy as np

ROI_TOP = 330                  # randul de la care incepe drumul; mai sus e cer si peisaj
TOPHAT_KERNEL_SIZE = (41, 5)   # (latime, inaltime) in pixeli; mai lat decat orice marcaj
TOPHAT_THRESHOLD = 25          # cu cat trebuie sa fie pixelul mai luminos decat fundalul local


def threshold_lane_pixels(rgb):
    """Intoarce o masca alb-negru: 255 unde pixelul pare marcaj, 0 in rest."""
    gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)

    # Top-hat = imaginea minus fundalul ei local. Raman doar structurile luminoase
    # mai inguste decat kernelul (marcajele); asfaltul, oricat de luminat, devine ~0.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, TOPHAT_KERNEL_SIZE)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    mask = (tophat > TOPHAT_THRESHOLD).astype(np.uint8) * 255
    mask[:ROI_TOP] = 0  # ignoram tot ce e deasupra drumului
    return mask


def main():
    for path in sys.argv[1:]:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"Nu pot citi imaginea: {path}")
            continue
        mask = threshold_lane_pixels(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        side_by_side = np.hstack([bgr, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow("Original | masca de marcaje", side_by_side)
        if cv2.waitKey(0) == 27:  # 27 = ESC
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
