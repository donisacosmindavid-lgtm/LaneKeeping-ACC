"""Pasul 3: detectia benzilor rulata live pe camera masinii.

Stanga: imaginea camerei, cu banda detectata in verde si eroarea laterala.
Dreapta: vederea de sus, cu ferestrele glisante si curbele potrivite.

Rulare (cu serverul CARLA pornit):
    venv\\Scripts\\python.exe lane_view.py
Iesire: ESC sau inchiderea ferestrei.
"""

import cv2
import numpy as np

from display import Window
from lane_detection import BEV_HEIGHT, BEV_WIDTH, detect_lane, draw_lane_overlay, lateral_offset
from simulation import FIXED_DELTA, IMAGE_HEIGHT, IMAGE_WIDTH, Simulation


def compose_view(rgb, left_fit, right_fit, debug):
    """Construieste imaginea afisata: camera cu banda detectata + vederea de sus."""
    offset = lateral_offset(left_fit, right_fit)

    camera_view = draw_lane_overlay(rgb, left_fit, right_fit)
    text = "Eroare laterala: fara detectie" if offset is None else f"Eroare laterala: {offset:+.2f} m"
    cv2.putText(camera_view, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    canvas = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH + BEV_WIDTH, 3), dtype=np.uint8)
    canvas[:, :IMAGE_WIDTH] = camera_view
    canvas[:BEV_HEIGHT, IMAGE_WIDTH:] = debug[:, :, ::-1]  # debug e in ordine BGR (OpenCV)
    return canvas


def main():
    window = Window(IMAGE_WIDTH + BEV_WIDTH, IMAGE_HEIGHT, "ADAS - detectia benzilor",
                    fps=1.0 / FIXED_DELTA)
    try:
        with Simulation() as sim:
            running = True
            while running:
                rgb = sim.tick()
                left_fit, right_fit, debug = detect_lane(rgb)
                running = window.show(compose_view(rgb, left_fit, right_fit, debug))
    finally:
        window.close()


if __name__ == "__main__":
    main()
