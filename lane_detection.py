"""Pasul 3: detectia benzilor cu computer vision clasic.

Etapa 1: filtrarea - din imaginea camerei pastram doar pixelii care seamana
         cu un marcaj rutier.
Etapa 2: transformarea de perspectiva - "indreptam" imaginea ca si cum am
         privi drumul de sus (bird's-eye view, prescurtat BEV).

Demo pe imagini salvate (o tasta trece la urmatoarea, ESC iese):
    venv\\Scripts\\python.exe lane_detection.py data\\frames\\frame_0060.png
"""

import math
import sys

import cv2
import numpy as np

from camera_view import CAMERA_FOV, CAMERA_TRANSFORM, IMAGE_HEIGHT, IMAGE_WIDTH

# --- Etapa 1: filtrarea ---
ROI_TOP = 330                  # randul de la care incepe drumul; mai sus e cer si peisaj
TOPHAT_KERNEL_SIZE = (41, 5)   # (latime, inaltime) in pixeli; mai lat decat orice marcaj
TOPHAT_THRESHOLD = 25          # cu cat trebuie sa fie pixelul mai luminos decat fundalul local

# --- Etapa 2: vederea de sus ---
CAMERA_HEIGHT = CAMERA_TRANSFORM.location.z   # inaltimea camerei deasupra soselei (m)
BEV_X_NEAR = 5.0               # de la cati metri in fata masinii incepe zona privita de sus
BEV_X_FAR = 25.0               # pana la cati metri in fata
BEV_HALF_WIDTH = 5.0           # cati metri vedem la stanga si la dreapta
BEV_WIDTH = 400                # dimensiunile imaginii de sus, in pixeli
BEV_HEIGHT = 400
XM_PER_PIX = 2 * BEV_HALF_WIDTH / BEV_WIDTH        # metri pe pixel, lateral
YM_PER_PIX = (BEV_X_FAR - BEV_X_NEAR) / BEV_HEIGHT  # metri pe pixel, inainte


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


def ground_to_pixel(forward, right):
    """Unde apare in imagine un punct de pe sosea aflat la `forward` metri in fata
    camerei si `right` metri la dreapta ei (modelul camerei pinhole, drum plat)."""
    # Distanta focala in pixeli: rezulta din latimea imaginii si unghiul de vedere
    focal = IMAGE_WIDTH / (2 * math.tan(math.radians(CAMERA_FOV) / 2))
    u = IMAGE_WIDTH / 2 + focal * right / forward            # coloana
    v = IMAGE_HEIGHT / 2 + focal * CAMERA_HEIGHT / forward   # rand
    return u, v


def bev_source_points():
    """Colturile dreptunghiului de pe sosea, asa cum se vad in imaginea camerei (un trapez)."""
    return np.float32([
        ground_to_pixel(BEV_X_FAR, -BEV_HALF_WIDTH),    # departe, stanga
        ground_to_pixel(BEV_X_FAR, BEV_HALF_WIDTH),     # departe, dreapta
        ground_to_pixel(BEV_X_NEAR, BEV_HALF_WIDTH),    # aproape, dreapta
        ground_to_pixel(BEV_X_NEAR, -BEV_HALF_WIDTH),   # aproape, stanga
    ])


def build_bev_matrix():
    """Matricea care duce trapezul din imaginea camerei in dreptunghiul vederii de sus."""
    destination = np.float32([
        [0, 0],
        [BEV_WIDTH, 0],
        [BEV_WIDTH, BEV_HEIGHT],
        [0, BEV_HEIGHT],
    ])
    return cv2.getPerspectiveTransform(bev_source_points(), destination)


BEV_MATRIX = build_bev_matrix()  # camera e fixa pe masina, deci o calculam o singura data


def warp_to_bev(image):
    """Transforma o imagine a camerei (sau o masca) in vedere de sus.
    In rezultat: masina e jos la mijloc, inainte = in sus, 1 pixel = XM_PER_PIX x YM_PER_PIX metri."""
    return cv2.warpPerspective(np.ascontiguousarray(image), BEV_MATRIX, (BEV_WIDTH, BEV_HEIGHT),
                               flags=cv2.INTER_NEAREST)


def main():
    for path in sys.argv[1:]:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"Nu pot citi imaginea: {path}")
            continue
        mask = threshold_lane_pixels(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # Desenam trapezul peste original, ca sa vedem ce zona de drum decupam
        outlined = bgr.copy()
        cv2.polylines(outlined, [np.int32(bev_source_points())], True, (0, 0, 255), 2)
        camera_row = np.hstack([outlined, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        bev_row = np.hstack([warp_to_bev(bgr), cv2.cvtColor(warp_to_bev(mask), cv2.COLOR_GRAY2BGR)])

        cv2.imshow("Camera: original cu trapez | masca de marcaje", camera_row)
        cv2.imshow("Vedere de sus: original | masca", bev_row)
        if cv2.waitKey(0) == 27:  # 27 = ESC
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
