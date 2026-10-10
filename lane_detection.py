"""Pasul 3: detectia benzilor cu computer vision clasic.

Etapa 1: filtrarea - din imaginea camerei pastram doar pixelii care seamana
         cu un marcaj rutier.
Etapa 2: transformarea de perspectiva - "indreptam" imaginea ca si cum am
         privi drumul de sus (bird's-eye view, prescurtat BEV).
Etapa 3: sliding window - gasim pixelii celor doua linii ale benzii noastre
         si potrivim cate o curba prin ei.
Etapa 4: eroarea laterala in metri si verificarea de plauzibilitate, ca detectia
         sa spuna "nu stiu" in loc sa raporteze valori absurde.
Etapa 5: urmarire intre cadre (clasa LaneDetector), ca ferestrele sa nu piarda
         liniile in curbe stranse.

Demo pe imagini salvate (o tasta trece la urmatoarea, ESC iese):
    venv\\Scripts\\python.exe lane_detection.py data\\frames\\frame_0060.png
"""

import math
import sys

import cv2
import numpy as np

from simulation import CAMERA_FOV, CAMERA_TRANSFORM, IMAGE_HEIGHT, IMAGE_WIDTH

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

# --- Eroarea laterala si verificarea de plauzibilitate ---
LANE_WIDTH = 3.5               # latimea asteptata a benzii (m)
LANE_WIDTH_TOLERANCE = 0.7     # cu cat se poate abate latimea masurata ca sa credem detectia (m)
MAX_FIT_RESIDUAL = 4.0         # abaterea maxima a pixelilor fata de curbele potrivite (pixeli)


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
BEV_MATRIX_INV = np.linalg.inv(BEV_MATRIX)  # drumul invers: din vederea de sus in imaginea camerei


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


def track_line(bev_mask, base_x=None, prior_fit=None):
    """Aduna pixelii unei linii cu ferestre glisante, de jos in sus.
    - cu base_x: porneste de la acea coloana si fiecare fereastra se recentreaza pe ce gaseste;
    - cu prior_fit: ferestrele urmeaza curba [a, b, c] gasita la cadrul anterior.
    Intoarce coordonatele pixelilor liniei (xs, ys) si ferestrele folosite (pentru desen)."""
    ys, xs = bev_mask.nonzero()  # coordonatele tuturor pixelilor albi
    window_height = BEV_HEIGHT // N_WINDOWS
    current_x = base_x
    selected = []
    windows = []

    for i in range(N_WINDOWS):
        y_high = BEV_HEIGHT - i * window_height  # fereastra 0 e cea de jos, langa masina
        y_low = y_high - window_height
        if prior_fit is not None:
            # Linia nu sare de la un cadru la altul: o cautam unde era acum 0.05 s
            current_x = int(np.polyval(prior_fit, (y_low + y_high) / 2))
        x_low = current_x - WINDOW_MARGIN
        x_high = current_x + WINDOW_MARGIN
        windows.append((x_low, y_low, x_high, y_high))

        inside = ((ys >= y_low) & (ys < y_high) & (xs >= x_low) & (xs < x_high)).nonzero()[0]
        selected.append(inside)
        # Fara curba anterioara: daca am prins destui pixeli, ne recentram pe media lor
        if prior_fit is None and len(inside) >= MIN_PIXELS:
            current_x = int(xs[inside].mean())

    selected = np.concatenate(selected)
    return xs[selected], ys[selected], windows


def fit_line(xs, ys):
    """Potriveste x = a*y^2 + b*y + c prin pixelii unei singure linii. Intoarce [a, b, c]."""
    # Cu pixeli doar dintr-o zona scurta (o singura liniuta) o parabola ar fi nesigura: dreapta
    if ys.max() - ys.min() > BEV_HEIGHT / 2:
        return np.polyfit(ys, xs, 2)
    return np.concatenate([[0.0], np.polyfit(ys, xs, 1)])


def fit_parallel_lines(left_pixels, right_pixels):
    """Potriveste ambele linii deodata. Fiind paralele, au aceeasi forma (a si b comune)
    si difera doar prin pozitie (c_left, c_right): x = a*y^2 + b*y + c.
    Intoarce (left_fit, right_fit, residual), unde residual este abaterea medie patratica
    a pixelilor fata de curbe, in pixeli."""
    (left_xs, left_ys), (right_xs, right_ys) = left_pixels, right_pixels
    xs = np.concatenate([left_xs, right_xs]).astype(float)
    ys = np.concatenate([left_ys, right_ys]).astype(float)
    is_left = np.concatenate([np.ones(len(left_xs)), np.zeros(len(right_xs))])

    # Fiecare pixel da o ecuatie: a*y^2 + b*y + c_left*[e pe stanga] + c_right*[e pe dreapta] = x
    columns = [ys, is_left, 1 - is_left]
    # Cu pixeli doar dintr-o zona scurta o parabola ar fi nesigura: atunci potrivim drepte
    use_parabola = ys.max() - ys.min() > BEV_HEIGHT / 2
    if use_parabola:
        columns.insert(0, ys ** 2)
    equations = np.column_stack(columns)

    # Cele mai mici patrate: coeficientii care fac suma abaterilor patratice minima
    coefficients = np.linalg.lstsq(equations, xs, rcond=None)[0]
    residual = np.sqrt(np.mean((equations @ coefficients - xs) ** 2))

    if not use_parabola:
        coefficients = np.concatenate([[0.0], coefficients])
    a, b, c_left, c_right = coefficients
    return np.array([a, b, c_left]), np.array([a, b, c_right]), residual


def fit_lane(left_pixels, right_pixels):
    """Din pixelii celor doua linii (fiecare (xs, ys) sau None) scoate curbele benzii.
    Intoarce (left_fit, right_fit, rejected): rejected e True daca am gasit ambele linii,
    dar rezultatul nu arata ca o banda si l-am respins."""
    if left_pixels is not None and right_pixels is not None:
        left_fit, right_fit, residual = fit_parallel_lines(left_pixels, right_pixels)
        if lane_is_plausible(left_fit, right_fit, residual):
            return left_fit, right_fit, False
        return None, None, True  # mai bine "nu stiu" decat o banda care nu poate exista
    if left_pixels is not None:
        return fit_line(*left_pixels), None, False
    if right_pixels is not None:
        return None, fit_line(*right_pixels), False
    return None, None, False


def lane_is_plausible(left_fit, right_fit, residual):
    """Verifica daca cele doua linii arata ca o banda: distanta dintre ele seamana cu
    latimea unei benzi, iar pixelii chiar stau pe doua curbe paralele."""
    width = (right_fit[2] - left_fit[2]) * XM_PER_PIX
    return abs(width - LANE_WIDTH) <= LANE_WIDTH_TOLERANCE and residual <= MAX_FIT_RESIDUAL


def find_lane_lines(bev_mask, prior=(None, None)):
    """Detecteaza cele doua linii ale benzii noastre in masca vazuta de sus.
    prior: curbele (stanga, dreapta) de la cadrul anterior, daca le avem; fara ele,
    punctul de start al fiecarei linii se cauta in histograma.
    Intoarce (left_fit, right_fit, debug): coeficientii [a, b, c] ai fiecarei linii
    (None daca linia nu a fost gasita) si o imagine color cu ce a facut algoritmul."""
    debug = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)
    pixels = []
    colors = [(0, 0, 255), (255, 0, 0)]  # stanga rosu, dreapta albastru (BGR)

    need_bases = any(fit is None for fit in prior)
    bases = find_line_bases(bev_mask) if need_bases else (None, None)
    for base_x, prior_fit, color in zip(bases, prior, colors):
        if prior_fit is not None:
            xs, ys, windows = track_line(bev_mask, prior_fit=prior_fit)
        elif base_x is not None:
            xs, ys, windows = track_line(bev_mask, base_x=base_x)
        else:
            pixels.append(None)
            continue
        pixels.append((xs, ys) if len(xs) >= MIN_FIT_PIXELS else None)

        debug[ys, xs] = color
        for x_low, y_low, x_high, y_high in windows:
            cv2.rectangle(debug, (x_low, y_low), (x_high, y_high), (0, 255, 0), 1)

    left_fit, right_fit, rejected = fit_lane(pixels[0], pixels[1])

    plot_y = np.arange(BEV_HEIGHT)
    for fit in (left_fit, right_fit):
        if fit is not None:
            points = np.int32(np.column_stack([np.polyval(fit, plot_y), plot_y]))
            cv2.polylines(debug, [points], False, (0, 255, 255), 2)
    if rejected:
        cv2.putText(debug, "respins", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return left_fit, right_fit, debug


def detect_lane(rgb):
    """Tot lantul, de la imaginea camerei la liniile benzii: filtrare, vedere de sus, sliding window.
    Intoarce (left_fit, right_fit, debug), ca find_lane_lines."""
    mask = threshold_lane_pixels(rgb)
    return find_lane_lines(warp_to_bev(mask))


class LaneDetector:
    """Detectia benzii cu urmarire intre cadre: ferestrele pornesc de la curbele gasite
    la cadrul anterior, deci nu pierd linia in golurile dintre liniute, nici in curbe."""

    def __init__(self):
        self._prior = (None, None)  # curbele (stanga, dreapta) de la cadrul anterior

    def update(self, rgb):
        """Primeste imaginea camerei si intoarce (left_fit, right_fit, debug)."""
        bev_mask = warp_to_bev(threshold_lane_pixels(rgb))
        left_fit, right_fit, debug = find_lane_lines(bev_mask, self._prior)
        # Daca detectia a esuat sau a fost respinsa, la cadrul urmator o luam de la zero
        self._prior = (left_fit, right_fit)
        return left_fit, right_fit, debug


def lateral_offset(left_fit, right_fit):
    """Cat de departe e masina de centrul benzii, in metri, masurat la BEV_X_NEAR metri
    in fata camerei. Pozitiv = masina e la dreapta centrului. None daca nu vedem nicio linie."""
    y = BEV_HEIGHT - 1  # randul de jos al vederii de sus, cel mai apropiat de masina
    half_lane = LANE_WIDTH / 2 / XM_PER_PIX  # jumatate de banda, in pixeli

    if left_fit is not None and right_fit is not None:
        lane_center = (np.polyval(left_fit, y) + np.polyval(right_fit, y)) / 2
    elif left_fit is not None:
        lane_center = np.polyval(left_fit, y) + half_lane
    elif right_fit is not None:
        lane_center = np.polyval(right_fit, y) - half_lane
    else:
        return None

    car_x = BEV_WIDTH / 2  # camera e pe axa masinii, deci masina e la mijlocul imaginii
    return (car_x - lane_center) * XM_PER_PIX


def heading_error(left_fit, right_fit):
    """Unghiul dintre directia benzii si directia masinii, in radiani, masurat in acelasi
    loc ca eroarea laterala. Pozitiv = banda o ia la dreapta fata de masina.
    None daca nu vedem nicio linie."""
    fits = [fit for fit in (left_fit, right_fit) if fit is not None]
    if not fits:
        return None

    y = BEV_HEIGHT - 1
    # Panta unei linii in pixeli: derivata lui x = a*y^2 + b*y + c, adica 2*a*y + b
    slopes_px = [2 * fit[0] * y + fit[1] for fit in fits]
    slope_px = sum(slopes_px) / len(slopes_px)  # media celor doua linii
    # In metri: un pixel lateral = XM_PER_PIX, un pixel inainte = YM_PER_PIX.
    # Minusul apare pentru ca "inainte" inseamna randuri tot mai mici in imagine.
    slope_m = -slope_px * XM_PER_PIX / YM_PER_PIX
    return math.atan(slope_m)


def lane_curvature(left_fit, right_fit):
    """Curbura benzii in 1/m (inversul razei curbei), pozitiva = curba la dreapta.
    None daca nu vedem nicio linie."""
    fits = [fit for fit in (left_fit, right_fit) if fit is not None]
    if not fits:
        return None

    a = sum(fit[0] for fit in fits) / len(fits)
    # Curbura este a doua derivata a liniei: pentru x = a*y^2 + b*y + c, ea este 2*a (in pixeli).
    # In metri: lateral inmultim cu XM_PER_PIX, iar inainte impartim de doua ori la YM_PER_PIX.
    # (Aproximare buna cat timp linia nu e foarte inclinata fata de directia masinii.)
    return 2 * a * XM_PER_PIX / YM_PER_PIX ** 2


def draw_lane_overlay(rgb, left_fit, right_fit):
    """Intoarce imaginea camerei cu banda detectata colorata in verde."""
    if left_fit is None or right_fit is None:
        return rgb.copy()

    # Desenam banda in vederea de sus, ca poligon intre cele doua curbe...
    ys = np.arange(BEV_HEIGHT)
    left_points = np.column_stack([np.polyval(left_fit, ys), ys])
    right_points = np.column_stack([np.polyval(right_fit, ys), ys])[::-1]
    lane = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
    cv2.fillPoly(lane, [np.int32(np.vstack([left_points, right_points]))], (0, 255, 0))

    # ...apoi o ducem inapoi in perspectiva camerei si o suprapunem semitransparent
    lane_in_camera = cv2.warpPerspective(lane, BEV_MATRIX_INV, (IMAGE_WIDTH, IMAGE_HEIGHT))
    return cv2.addWeighted(rgb, 1.0, lane_in_camera, 0.3, 0)


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
