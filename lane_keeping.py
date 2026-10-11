"""Pasii 4 si 5: masina tine singura banda (lane keeping) si distanta fata de
masina din fata (adaptive cruise control, ACC).

Detectia benzii da eroarea laterala, eroarea de orientare si curbura benzii.
Regulatorul Stanley transforma primele doua in unghi de volan. Un regulator PI
tine viteza ceruta, care este cea mai mica dintre: viteza de croaziera, viteza
permisa de curba si viteza ceruta de ACC din masuratorile radarului. Daca detectia
benzii lipseste prea mult timp, masina se opreste. Pilotul automat al CARLA este oprit.

Rulare (cu serverul CARLA pornit):
    venv\\Scripts\\python.exe lane_keeping.py
    venv\\Scripts\\python.exe lane_keeping.py --log results\\test.csv --steps 1200
Iesire: ESC sau inchiderea ferestrei.
"""

import argparse
import math
import time

import cv2

from controllers import AccController, SpeedController, StanleyController
from display import Window
from lane_detection import (BEV_HEIGHT, BEV_WIDTH, BEV_X_NEAR, LaneDetector, heading_error,
                            lane_center_lateral, lane_curvature, lateral_offset)
from lane_view import compose_view
from metrics import MetricsLogger
from radar import LeadTracker
from vehicle_detection import VehicleDetector, draw_vehicles
from simulation import (CAMERA_TRANSFORM, FIXED_DELTA, IMAGE_HEIGHT, IMAGE_WIDTH, RADAR_TRANSFORM,
                        REAR_AXLE_OFFSET, WHEELBASE, Simulation)

CRUISE_SPEED = 50 / 3.6       # viteza de croaziera, pe drum drept (m/s)
STANLEY_GAIN = 3.0            # k din formula Stanley
SPEED_KP = 0.3                # castigul proportional al regulatorului de viteza
SPEED_KI = 0.05               # castigul integral
MAX_THROTTLE = 0.6            # acceleratia maxima ceruta (din 1); 1.0 ar insemna ~6 m/s^2
MAX_BRAKE = 0.3               # frana maxima ceruta (din 1); in CARLA, 1.0 opreste cu ~3 g
MAX_LATERAL_ACCEL = 2.0       # acceleratia laterala maxima acceptata in curbe (m/s^2)
CURVATURE_RISE_TIME = 0.3     # constanta de timp a filtrului de curbura cand curbura creste (s)
CURVATURE_FALL_TIME = 3.0     # ...si cand scade (s)
MAX_BLIND_FRAMES = 10         # dupa atatea cadre la rand fara detectie, masina franeaza (0.5 s)
CURVATURE_LAG_TIME = 1.5      # intarzierea curburii folosite la corectia directiei (s)
ACC_TIME_GAP = 1.8            # intervalul de timp pastrat fata de masina din fata (s)
ACC_STANDSTILL_DISTANCE = 6.0 # distanta pastrata cand ambele masini stau pe loc (m)
ACC_GAIN = 0.3                # m/s de corectie pentru fiecare metru de eroare de distanta (1/s)
# Masina din fata, in demonstratia din main()
LEAD_START_DISTANCE = 35.0    # la cati metri in fata noastra porneste (m)
LEAD_FAST_SPEED = 40 / 3.6    # alterneaza intre aceste doua viteze (m/s)...
LEAD_SLOW_SPEED = 25 / 3.6
LEAD_PHASE_TIME = 20.0        # ...cate atatea secunde pe fiecare
# Distanta de la puntea spate la punctul in care detectia masoara erorile (m)
LOOKAHEAD = REAR_AXLE_OFFSET + CAMERA_TRANSFORM.location.x + BEV_X_NEAR


class LaneKeeper:
    """Un pas de reglare: imagine -> masuratori -> comenzi catre masina."""

    def __init__(self, sim):
        self.sim = sim
        self.detector = LaneDetector()
        self.lead_tracker = LeadTracker()
        self.lead = None                   # masina din fata vazuta de radar, sau None
        self.last_fits = (None, None)      # curbele benzii de la cadrul anterior
        self.stanley = StanleyController(STANLEY_GAIN, WHEELBASE)
        self.speed_controller = SpeedController(SPEED_KP, SPEED_KI, FIXED_DELTA,
                                                MAX_THROTTLE, MAX_BRAKE)
        self.acc = AccController(ACC_TIME_GAP, ACC_STANDSTILL_DISTANCE, ACC_GAIN)
        self.steer_angle = 0.0             # ultima comanda de directie (rad)
        self.curvature = 0.0               # curbura benzii din fata, filtrata; alege viteza (1/m)
        self.curvature_here = 0.0          # curbura estimata sub masina; corecteaza directia (1/m)
        self.target_speed = CRUISE_SPEED   # viteza ceruta in acest pas (m/s)
        self.frames_without_detection = 0

    def step(self, rgb):
        """Calculeaza si trimite comenzile pentru imaginea curenta.
        Intoarce (left_fit, right_fit, debug, offset, heading) pentru afisare."""
        speed = self.sim.speed()
        # Radarul: masina din fata, ca (distanta, viteza relativa, pozitie laterala),
        # sau None daca drumul e liber
        self.lead = self.lead_tracker.update(self.sim.radar_points, speed, self.lane_center_ahead)

        # Camera: banda. Ii spunem unde e masina din fata, ca sa n-o ia drept marcaj.
        left_fit, right_fit, debug = self.detector.update(rgb, self.lead)
        self.last_fits = (left_fit, right_fit)
        offset = lateral_offset(left_fit, right_fit)
        heading = heading_error(left_fit, right_fit)

        if offset is not None:
            self.frames_without_detection = 0
            self.update_curvature(lane_curvature(left_fit, right_fit))
            offset_here, heading_here = self.remove_curve_bias(offset, heading)
            self.steer_angle = self.stanley.compute(offset_here, heading_here, speed,
                                                    self.curvature_here)
        else:
            # Fara detectie pastram ultima comanda de directie si numaram cadrele
            self.frames_without_detection += 1

        self.target_speed = self.choose_target_speed(speed)
        throttle, brake = self.speed_controller.compute(self.target_speed, speed)
        self.sim.apply_control(self.steer_angle, throttle, brake)
        return left_fit, right_fit, debug, offset, heading

    def lane_center_ahead(self, distance):
        """Unde e centrul benzii la `distance` metri in fata radarului (m lateral, pozitiv =
        dreapta). Camera ajuta radarul: folosim curbele benzii de la cadrul anterior."""
        from_camera = distance + RADAR_TRANSFORM.location.x - CAMERA_TRANSFORM.location.x
        center = lane_center_lateral(*self.last_fits, from_camera)
        if center is None:
            # Fara detectie de banda, presupunem o curba de curbura constanta
            center = self.curvature * distance ** 2 / 2
        return center

    def update_curvature(self, measured):
        """Filtru trece-jos de ordinul 1 pe curbura masurata, cu doua constante de timp.
        Curbura o masuram in zona de 6-26 m din fata masinii: cand creste, incetinim
        repede; cand scade, masina e inca in curba, deci o lasam sa scada lent."""
        rising = abs(measured) > abs(self.curvature)
        time_constant = CURVATURE_RISE_TIME if rising else CURVATURE_FALL_TIME
        alpha = FIXED_DELTA / (time_constant + FIXED_DELTA)
        self.curvature += alpha * (measured - self.curvature)

        # Aceeasi masuratoare, intarziata: curbura vazuta acum in fata ajunge sub masina
        # abia dupa ce parcurgem distanta pana acolo
        alpha = FIXED_DELTA / (CURVATURE_LAG_TIME + FIXED_DELTA)
        self.curvature_here += alpha * (measured - self.curvature_here)

    def remove_curve_bias(self, offset, heading):
        """Scoate din masuratori partea care se datoreaza doar curbei.
        Intr-o curba, chiar si o masina perfect centrata vede, la LOOKAHEAD metri in fata,
        banda rotita cu curbura * LOOKAHEAD si deplasata lateral. Scazand aceste valori,
        erorile raman zero cand masina e pe centrul benzii."""
        heading_here = heading - self.curvature_here * LOOKAHEAD
        offset_here = offset + self.curvature_here * (LOOKAHEAD ** 2 - REAR_AXLE_OFFSET ** 2) / 2
        return offset_here, heading_here

    def choose_target_speed(self, speed):
        """Viteza ceruta: cea mai mica dintre viteza de croaziera, cea permisa de curba si
        cea ceruta de ACC. Zero daca nu mai vedem banda."""
        if self.frames_without_detection > MAX_BLIND_FRAMES:
            return 0.0
        # Intr-o curba, acceleratia laterala este v^2 * curbura. O limitam la
        # MAX_LATERAL_ACCEL, deci viteza maxima in curba este radical(a_max / curbura).
        curve_speed = math.sqrt(MAX_LATERAL_ACCEL / max(abs(self.curvature), 1e-6))
        target = min(CRUISE_SPEED, curve_speed)
        if self.lead is not None:
            distance, relative_speed, _ = self.lead
            target = min(target, self.acc.target_speed(distance, relative_speed, speed))
        return target


def draw_telemetry(canvas, keeper, heading):
    """Scrie viteza, distanta pana la masina din fata, orientarea si comanda de volan
    sub vederea de sus."""
    speed = keeper.sim.speed()
    if keeper.lead is None:
        lead_line = "Radar: drum liber"
    else:
        lead_line = f"Radar: {keeper.lead[0]:.0f} m (dorit {keeper.acc.desired_distance(speed):.0f})"
    lines = [
        f"Viteza: {speed * 3.6:.0f} km/h (tinta {keeper.target_speed * 3.6:.0f})",
        lead_line,
        "Orientare: -" if heading is None else f"Orientare: {math.degrees(heading):+.1f} grd",
        f"Volan: {math.degrees(keeper.steer_angle):+.1f} grd",
    ]
    for i, line in enumerate(lines):
        position = (IMAGE_WIDTH + 15, BEV_HEIGHT + 40 + 38 * i)
        cv2.putText(canvas, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def parse_arguments():
    """Optiunile din linia de comanda."""
    parser = argparse.ArgumentParser(description="Lane keeping si ACC in CARLA.")
    parser.add_argument("--log", metavar="FISIER.csv", help="scrie metricile fiecarui cadru in acest fisier")
    parser.add_argument("--steps", type=int, metavar="N", help="opreste rularea dupa N pasi de simulare")
    return parser.parse_args()


def log_metrics(logger, sim, keeper, step, offset, vehicles, latency_ms):
    """Scrie un rand de metrici: ce am masurat noi si, alaturi, adevarul din simulator."""
    speed = sim.speed()
    if keeper.lead is None:
        lead_distance = lead_rel_speed = None
    else:
        lead_distance, lead_rel_speed, _ = keeper.lead
    if sim.lead is None:
        true_distance = true_rel_speed = None
    else:
        true_distance, true_rel_speed = sim.true_lead_gap()

    logger.log(
        step=step,
        time_s=step * FIXED_DELTA,
        speed_kmh=speed * 3.6,
        target_speed_kmh=keeper.target_speed * 3.6,
        lane_detected=int(offset is not None),
        lateral_error_m=offset,
        lateral_error_true_m=sim.true_lateral_error(),
        steer_deg=math.degrees(keeper.steer_angle),
        lead_detected=int(keeper.lead is not None),
        lead_distance_m=lead_distance,
        lead_distance_true_m=true_distance,
        lead_rel_speed_ms=lead_rel_speed,
        lead_rel_speed_true_ms=true_rel_speed,
        desired_distance_m=keeper.acc.desired_distance(speed),
        vehicles_seen=len(vehicles),
        latency_ms=latency_ms,
    )


def main():
    arguments = parse_arguments()
    window = Window(IMAGE_WIDTH + BEV_WIDTH, IMAGE_HEIGHT, "ADAS - lane keeping si ACC",
                    fps=1.0 / FIXED_DELTA)
    logger = MetricsLogger(arguments.log) if arguments.log else None
    try:
        with Simulation(autopilot=False, lead_distance=LEAD_START_DISTANCE) as sim:
            keeper = LaneKeeper(sim)
            vehicle_detector = VehicleDetector()
            running = True
            step = 0
            while running:
                # Masina din fata alterneaza intre doua viteze, ca sa vedem ACC-ul la lucru
                step += 1
                slow = (step * FIXED_DELTA) % (2 * LEAD_PHASE_TIME) >= LEAD_PHASE_TIME
                sim.set_lead_speed(LEAD_SLOW_SPEED if slow else LEAD_FAST_SPEED)
                rgb = sim.tick()

                # Latenta: cat dureaza tot ce facem cu un cadru, de la imagine la comenzi
                started = time.perf_counter()
                left_fit, right_fit, debug, offset, heading = keeper.step(rgb)
                vehicles = vehicle_detector.detect(rgb)
                latency_ms = 1000 * (time.perf_counter() - started)
                if logger is not None:
                    log_metrics(logger, sim, keeper, step, offset, vehicles, latency_ms)

                canvas = compose_view(rgb, left_fit, right_fit, debug)
                draw_telemetry(canvas, keeper, heading)
                # YOLO e doar pentru afisare: imaginea camerei e in stanga, deci
                # dreptunghiurile se deseneaza direct, la aceleasi coordonate
                draw_vehicles(canvas, vehicles)
                running = window.show(canvas)
                if arguments.steps is not None and step >= arguments.steps:
                    running = False
    finally:
        if logger is not None:
            logger.close()
        window.close()


if __name__ == "__main__":
    main()
