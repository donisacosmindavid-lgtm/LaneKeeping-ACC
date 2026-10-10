"""Pasul 4: masina tine singura banda (lane keeping).

Detectia benzii da eroarea laterala, eroarea de orientare si curbura benzii.
Regulatorul Stanley transforma primele doua in unghi de volan. Un regulator PI
tine viteza: cea de croaziera pe drum drept, mai mica in curbe stranse si zero
daca detectia benzii lipseste prea mult timp. Pilotul automat al CARLA este oprit.

Rulare (cu serverul CARLA pornit):
    venv\\Scripts\\python.exe lane_keeping.py
Iesire: ESC sau inchiderea ferestrei.
"""

import math

import cv2

from controllers import SpeedController, StanleyController
from display import Window
from lane_detection import (BEV_HEIGHT, BEV_WIDTH, LaneDetector, heading_error, lane_curvature,
                            lateral_offset)
from lane_view import compose_view
from simulation import FIXED_DELTA, IMAGE_HEIGHT, IMAGE_WIDTH, Simulation

CRUISE_SPEED = 50 / 3.6       # viteza de croaziera, pe drum drept (m/s)
STANLEY_GAIN = 3.0            # k din formula Stanley
SPEED_KP = 0.3                # castigul proportional al regulatorului de viteza
SPEED_KI = 0.05               # castigul integral
MAX_LATERAL_ACCEL = 2.0       # acceleratia laterala maxima acceptata in curbe (m/s^2)
CURVATURE_RISE_TIME = 0.3     # constanta de timp a filtrului de curbura cand curbura creste (s)
CURVATURE_FALL_TIME = 3.0     # ...si cand scade (s)
MAX_BLIND_FRAMES = 10         # dupa atatea cadre la rand fara detectie, masina franeaza (0.5 s)


class LaneKeeper:
    """Un pas de reglare: imagine -> masuratori -> comenzi catre masina."""

    def __init__(self, sim):
        self.sim = sim
        self.detector = LaneDetector()
        self.stanley = StanleyController(STANLEY_GAIN)
        self.speed_controller = SpeedController(SPEED_KP, SPEED_KI, FIXED_DELTA)
        self.steer_angle = 0.0             # ultima comanda de directie (rad)
        self.curvature = 0.0               # curbura benzii, filtrata (1/m)
        self.target_speed = CRUISE_SPEED   # viteza ceruta in acest pas (m/s)
        self.frames_without_detection = 0

    def step(self, rgb):
        """Calculeaza si trimite comenzile pentru imaginea curenta.
        Intoarce (left_fit, right_fit, debug, offset, heading) pentru afisare."""
        left_fit, right_fit, debug = self.detector.update(rgb)
        offset = lateral_offset(left_fit, right_fit)
        heading = heading_error(left_fit, right_fit)
        speed = self.sim.speed()

        if offset is not None:
            self.frames_without_detection = 0
            self.steer_angle = self.stanley.compute(offset, heading, speed)
            self.update_curvature(lane_curvature(left_fit, right_fit))
        else:
            # Fara detectie pastram ultima comanda de directie si numaram cadrele
            self.frames_without_detection += 1

        self.target_speed = self.choose_target_speed()
        throttle, brake = self.speed_controller.compute(self.target_speed, speed)
        self.sim.apply_control(self.steer_angle, throttle, brake)
        return left_fit, right_fit, debug, offset, heading

    def update_curvature(self, measured):
        """Filtru trece-jos de ordinul 1 pe curbura masurata, cu doua constante de timp.
        Curbura o masuram in zona de 6-26 m din fata masinii: cand creste, incetinim
        repede; cand scade, masina e inca in curba, deci o lasam sa scada lent."""
        rising = abs(measured) > abs(self.curvature)
        time_constant = CURVATURE_RISE_TIME if rising else CURVATURE_FALL_TIME
        alpha = FIXED_DELTA / (time_constant + FIXED_DELTA)
        self.curvature += alpha * (measured - self.curvature)

    def choose_target_speed(self):
        """Viteza ceruta: de croaziera, redusa in curbe, zero daca nu mai vedem banda."""
        if self.frames_without_detection > MAX_BLIND_FRAMES:
            return 0.0
        # Intr-o curba, acceleratia laterala este v^2 * curbura. O limitam la
        # MAX_LATERAL_ACCEL, deci viteza maxima in curba este radical(a_max / curbura).
        curve_speed = math.sqrt(MAX_LATERAL_ACCEL / max(abs(self.curvature), 1e-6))
        return min(CRUISE_SPEED, curve_speed)


def draw_telemetry(canvas, keeper, heading):
    """Scrie viteza, orientarea si comanda de volan sub vederea de sus."""
    lines = [
        f"Viteza: {keeper.sim.speed() * 3.6:.0f} km/h (tinta {keeper.target_speed * 3.6:.0f})",
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
