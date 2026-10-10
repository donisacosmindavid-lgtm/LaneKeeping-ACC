"""Pasul 4: masina tine singura banda (lane keeping).

Detectia benzii da eroarea laterala si eroarea de orientare; regulatorul Stanley
le transforma in unghi de volan, iar un regulator PI tine viteza constanta.
Pilotul automat al CARLA este oprit.

Rulare (cu serverul CARLA pornit):
    venv\\Scripts\\python.exe lane_keeping.py
Iesire: ESC sau inchiderea ferestrei.
"""

import math

import cv2

from controllers import SpeedController, StanleyController
from display import Window
from lane_detection import BEV_HEIGHT, BEV_WIDTH, LaneDetector, heading_error, lateral_offset
from lane_view import compose_view
from simulation import FIXED_DELTA, IMAGE_HEIGHT, IMAGE_WIDTH, Simulation

TARGET_SPEED = 50 / 3.6   # viteza tinta in m/s (50 km/h)
STANLEY_GAIN = 3.0        # k din formula Stanley
SPEED_KP = 0.3            # castigul proportional al regulatorului de viteza
SPEED_KI = 0.05           # castigul integral


class LaneKeeper:
    """Un pas de reglare: imagine -> masuratori -> comenzi catre masina."""

    def __init__(self, sim):
        self.sim = sim
        self.detector = LaneDetector()
        self.stanley = StanleyController(STANLEY_GAIN)
        self.speed_controller = SpeedController(SPEED_KP, SPEED_KI, FIXED_DELTA)
        self.steer_angle = 0.0  # ultima comanda de directie (rad)

    def step(self, rgb):
        """Calculeaza si trimite comenzile pentru imaginea curenta.
        Intoarce (left_fit, right_fit, debug, offset, heading) pentru afisare."""
        left_fit, right_fit, debug = self.detector.update(rgb)
        offset = lateral_offset(left_fit, right_fit)
        heading = heading_error(left_fit, right_fit)
        speed = self.sim.speed()

        # Fara detectie pastram ultima comanda de directie
        if offset is not None:
            self.steer_angle = self.stanley.compute(offset, heading, speed)

        throttle, brake = self.speed_controller.compute(TARGET_SPEED, speed)
        self.sim.apply_control(self.steer_angle, throttle, brake)
        return left_fit, right_fit, debug, offset, heading


def draw_telemetry(canvas, keeper, heading):
    """Scrie viteza, orientarea si comanda de volan sub vederea de sus."""
    lines = [
        f"Viteza: {keeper.sim.speed() * 3.6:.0f} km/h (tinta {TARGET_SPEED * 3.6:.0f})",
        "Orientare: -" if heading is None else f"Orientare: {math.degrees(heading):+.1f} grd",
        f"Volan: {math.degrees(keeper.steer_angle):+.1f} grd",
    ]
    for i, line in enumerate(lines):
        position = (IMAGE_WIDTH + 15, BEV_HEIGHT + 45 + 40 * i)
        cv2.putText(canvas, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    window = Window(IMAGE_WIDTH + BEV_WIDTH, IMAGE_HEIGHT, "ADAS - lane keeping",
                    fps=1.0 / FIXED_DELTA)
    try:
        with Simulation(autopilot=False) as sim:
            keeper = LaneKeeper(sim)
            running = True
            while running:
                rgb = sim.tick()
                left_fit, right_fit, debug, offset, heading = keeper.step(rgb)
                canvas = compose_view(rgb, left_fit, right_fit, debug)
                draw_telemetry(canvas, keeper, heading)
                running = window.show(canvas)
    finally:
        window.close()


if __name__ == "__main__":
    main()
