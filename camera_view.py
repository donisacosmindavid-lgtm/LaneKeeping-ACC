"""Pasul 2: masina pe Town04 cu o camera frontala, imaginea afisata intr-o fereastra.

Rulare (cu serverul CARLA pornit):
    venv\\Scripts\\python.exe camera_view.py
Iesire: ESC sau inchiderea ferestrei.
"""

from display import Window
from simulation import FIXED_DELTA, IMAGE_HEIGHT, IMAGE_WIDTH, Simulation


def main():
    window = Window(IMAGE_WIDTH, IMAGE_HEIGHT, "ADAS - camera frontala", fps=1.0 / FIXED_DELTA)
    try:
        with Simulation() as sim:
            running = True
            while running:
                rgb = sim.tick()
                running = window.show(rgb)
    finally:
        window.close()


if __name__ == "__main__":
    main()
