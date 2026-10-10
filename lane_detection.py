"""Pasul 3: detectia benzilor cu computer vision clasic.

Etapa 1: filtrarea - din imaginea camerei pastram doar pixelii care seamana
         cu un marcaj rutier.
Etapa 2: transformarea de perspectiva - "indreptam" imaginea ca si cum am
         privi drumul de sus (bird's-eye view, prescurtat BEV).
Etapa 3: sliding window - gasim pixelii celor doua linii ale benzii noastre
         si potrivim cate o curba prin ei.

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

# --- Etapa 3: sliding window ---
HIST_SMOOTH = 15               # pe cate coloane vecine insumam histograma
MIN_PEAK = 50                  # cati pixeli albi trebuie sa aiba un varf ca sa fie considerat linie
N_WINDOWS = 10                 # cate ferestre suprapunem pe inaltimea imaginii
WINDOW_MARGIN = 40             # jumatate din latimea unei ferestre, in pixeli (1 m)
MIN_PIXELS = 15                # cati pixeli trebuie sa prinda o fereastra ca sa se recentreze
MIN_FIT_PIXELS = 60            # sub atatia pixeli nu avem incredere in linie


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


def find_line_bases(bev_mask):
    """Gaseste coloanele de start ale liniei din stanga si din dreapta masinii.
    Intoarce (left_x, right_x); None acolo unde nu exista nicio linie."""
    middle = BEV_WIDTH // 2  # masina e la mijlocul imaginii
    sides = [
        range(middle - 1, -1, -1),   # stanga: de la centru spre marginea stanga
        range(middle, BEV_WIDTH),    # dreapta: de la centru spre marginea dreapta
    ]
    # Cautam intai in jumatatea de jos (langa masina, unde pozitia conteaza cel mai mult).
    # Daca acolo e un gol intre doua liniute, cautam in toata imaginea.
    regions = [bev_mask[BEV_HEIGHT // 2:], bev_mask]

    bases = []
    for columns in sides:
        base = None
        for region in regions:
            base = _first_peak(_column_histogram(region), columns)
            if base is not None:
                break
        bases.append(base)
    return bases[0], bases[1]


def _column_histogram(mask):
    """Cati pixeli albi are fiecare coloana, insumat pe cateva coloane vecine."""
    histogram = (mask > 0).sum(axis=0)
    return np.convolve(histogram, np.ones(HIST_SMOOTH), mode="same")


def _first_peak(histogram, columns):
    """Parcurge coloanele in ordinea data si intoarce maximul primului varf intalnit."""
    best = None
    for x in columns:
        if histogram[x] >= MIN_PEAK:
            if best is None or histogram[x] > histogram[best]:
                best = x
        elif best is not None:
            break  # am trecut de primul varf; restul sunt linii ale altor benzi
    return best


def track_line(bev_mask, base_x):
    """Urmareste o linie de jos in sus cu ferestre glisante, pornind de la coloana base_x.
    Intoarce coordonatele pixelilor liniei (xs, ys) si ferestrele folosite (pentru desen)."""
    ys, xs = bev_mask.nonzero()  # coordonatele tuturor pixelilor albi
    window_height = BEV_HEIGHT // N_WINDOWS
    current_x = base_x
    selected = []
    windows = []

    for i in range(N_WINDOWS):
        y_high = BEV_HEIGHT - i * window_height  # fereastra 0 e cea de jos, langa masina
        y_low = y_high - window_height
        x_low = current_x - WINDOW_MARGIN
        x_high = current_x + WINDOW_MARGIN
        windows.append((x_low, y_low, x_high, y_high))

        inside = ((ys >= y_low) & (ys < y_high) & (xs >= x_low) & (xs < x_high)).nonzero()[0]
        selected.append(inside)
        # Daca am prins destui pixeli, urmatoarea fereastra se centreaza pe media lor
        if len(inside) >= MIN_PIXELS:
            current_x = int(xs[inside].mean())

    selected = np.concatenate(selected)
    return xs[selected], ys[selected], windows


def fit_line(xs, ys):
    """Potriveste x = a*y^2 + b*y + c prin pixelii unei linii. Intoarce [a, b, c] sau None."""
    if len(xs) < MIN_FIT_PIXELS:
        return None
    # Cu pixeli doar dintr-o zona scurta (o singura liniuta) o parabola ar fi nesigura: dreapta
    if ys.max() - ys.min() > BEV_HEIGHT / 2:
        return np.polyfit(ys, xs, 2)
    return np.concatenate([[0.0], np.polyfit(ys, xs, 1)])


def find_lane_lines(bev_mask):
    """Detecteaza cele doua linii ale benzii noastre in masca vazuta de sus.
    Intoarce (left_fit, right_fit, debug): coeficientii [a, b, c] ai fiecarei linii
    (None daca linia nu a fost gasita) si o imagine color cu ce a facut algoritmul."""
    debug = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)
    fits = []
    colors = [(0, 0, 255), (255, 0, 0)]  # stanga rosu, dreapta albastru (BGR)

    for base_x, color in zip(find_line_bases(bev_mask), colors):
        if base_x is None:
            fits.append(None)
            continue
        xs, ys, windows = track_line(bev_mask, base_x)
        fit = fit_line(xs, ys)
        fits.append(fit)

        debug[ys, xs] = color
        for x_low, y_low, x_high, y_high in windows:
            cv2.rectangle(debug, (x_low, y_low), (x_high, y_high), (0, 255, 0), 1)
        if fit is not None:
            plot_y = np.arange(BEV_HEIGHT)
            plot_x = np.polyval(fit, plot_y)
            points = np.int32(np.column_stack([plot_x, plot_y]))
            cv2.polylines(debug, [points], False, (0, 255, 255), 2)

    return fits[0], fits[1], debug


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
        left_fit, right_fit, debug = find_lane_lines(warp_to_bev(mask))
        bev_row = np.hstack([warp_to_bev(bgr), debug])
        print(f"{path}: stanga={left_fit}, dreapta={right_fit}")

        cv2.imshow("Camera: original cu trapez | masca de marcaje", camera_row)
        cv2.imshow("Vedere de sus: original | linii detectate", bev_row)
        if cv2.waitKey(0) == 27:  # 27 = ESC
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
