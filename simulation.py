"""Legatura cu simulatorul CARLA: harta, modul sincron, masina noastra cu camera
frontala si radar si, optional, o masina in fata pe aceeasi banda.

Folosire:
    with Simulation() as sim:
        rgb = sim.tick()   # avanseaza simularea cu un pas si intoarce imaginea camerei
"""

import math
import queue

import carla
import numpy as np

HOST = "localhost"
PORT = 2000
TM_PORT = 8000            # portul Traffic Manager-ului (pilotul automat din CARLA)
MAP_NAME = "Town04"
FIXED_DELTA = 0.05        # pasul de simulare in secunde -> 20 de pasi pe secunda
SPAWN_INDEX = 13          # punct de pornire pe autostrada, banda a doua
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
CAMERA_FOV = 90           # unghiul de vedere orizontal, in grade
# Pozitia camerei fata de centrul masinii: x = in fata, z = in sus (metri)
CAMERA_TRANSFORM = carla.Transform(carla.Location(x=1.5, z=2.4))
# Geometria masinii (Tesla Model 3), citita din simulator
WHEELBASE = 3.0           # ampatamentul: distanta dintre puntea fata si puntea spate (m)
REAR_AXLE_OFFSET = 1.39   # cat de in spatele centrului masinii este puntea spate (m)
# Radarul: montat la bara din fata
RADAR_TRANSFORM = carla.Transform(carla.Location(x=2.3, z=0.9))
RADAR_HORIZONTAL_FOV = 45       # unghiul de vedere orizontal, in grade; lat, ca sa vada si in curbe
RADAR_VERTICAL_FOV = 3          # ingust pe verticala, ca sa prinda cat mai putin asfalt
RADAR_RANGE = 100               # pana la ce distanta vede (m)
RADAR_POINTS_PER_SECOND = 4500


def image_to_rgb(image):
    """Transforma imaginea CARLA intr-o matrice numpy (inaltime, latime, 3) in ordine RGB."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # CARLA trimite B, G, R, A
    # Scoatem A, inversam in R, G, B si facem o copie proprie (independenta de memoria CARLA)
    return np.ascontiguousarray(array[:, :, :3][:, :, ::-1])


def radar_to_points(measurement):
    """Transforma masuratoarea radarului intr-o matrice numpy cu cate un rand pe punct detectat:
    [viteza (m/s, pozitiva = se departeaza), azimut (rad, pozitiv = dreapta),
     altitudine (rad, pozitiva = in sus), distanta (m)]."""
    points = np.frombuffer(measurement.raw_data, dtype=np.float32)
    return points.reshape((-1, 4)).copy()


class Simulation:
    """Masina noastra pe autostrada din Town04, cu camera frontala si radar, in mod sincron.
    Cu lead_distance, pune si o masina in fata, la atatia metri, pe aceeasi banda."""

    def __init__(self, autopilot=True, lead_distance=None):
        self.client = carla.Client(HOST, PORT)
        self.client.set_timeout(120.0)  # incarcarea unei harti poate dura mult

        self.world = self.client.get_world()
        if MAP_NAME not in self.world.get_map().name:
            self.world = self.client.load_world(MAP_NAME)

        self._original_settings = self.world.get_settings()
        self._traffic_manager = self.client.get_trafficmanager(TM_PORT)
        self._actors = []
        self._autopilot_vehicles = []          # masinile conduse de pilotul automat al CARLA
        self._image_queue = queue.Queue()
        self._radar_queue = queue.Queue()
        self.vehicle = None
        self.camera = None
        self.radar = None
        self.lead = None                       # masina din fata, daca exista
        self.radar_points = np.zeros((0, 4))   # punctele radarului din ultimul pas
        self.max_steer_angle = None            # unghiul maxim al rotilor din fata (rad)

        try:
            self._setup(autopilot, lead_distance)
        except Exception:
            self.close()  # nu lasam serverul in mod sincron daca pornirea esueaza la jumatate
            raise

    def _setup(self, autopilot, lead_distance):
        # Mod sincron: serverul calculeaza un pas doar cand ii cerem noi, cu world.tick()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA
        self.world.apply_settings(settings)
        self._traffic_manager.set_synchronous_mode(True)

        blueprints = self.world.get_blueprint_library()
        self._map = self.world.get_map()

        vehicle_bp = blueprints.find("vehicle.tesla.model3")
        vehicle_bp.set_attribute("role_name", "hero")
        spawn_point = self._map.get_spawn_points()[SPAWN_INDEX]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self._actors.append(self.vehicle)
        physics = self.vehicle.get_physics_control()
        self.max_steer_angle = math.radians(physics.wheels[0].max_steer_angle)
        # CARLA micsoreaza singura directia cand viteza creste, dupa aceasta curba
        # (viteza in km/h -> fractiune din unghiul cerut); o tinem minte ca s-o compensam
        self._steer_curve_speeds = [point.x for point in physics.steering_curve]
        self._steer_curve_factors = [point.y for point in physics.steering_curve]
        if autopilot:
            self.vehicle.set_autopilot(True, TM_PORT)
            self._autopilot_vehicles.append(self.vehicle)
            # Pilotul automat ramane pe banda lui, ca sa avem un test curat pentru detectie
            self._traffic_manager.auto_lane_change(self.vehicle, False)

        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
        camera_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
        camera_bp.set_attribute("fov", str(CAMERA_FOV))
        self.camera = self.world.spawn_actor(camera_bp, CAMERA_TRANSFORM, attach_to=self.vehicle)
        self._actors.append(self.camera)

        radar_bp = blueprints.find("sensor.other.radar")
        radar_bp.set_attribute("horizontal_fov", str(RADAR_HORIZONTAL_FOV))
        radar_bp.set_attribute("vertical_fov", str(RADAR_VERTICAL_FOV))
        radar_bp.set_attribute("range", str(RADAR_RANGE))
        radar_bp.set_attribute("points_per_second", str(RADAR_POINTS_PER_SECOND))
        self.radar = self.world.spawn_actor(radar_bp, RADAR_TRANSFORM, attach_to=self.vehicle)
        self._actors.append(self.radar)

        # Senzorii livreaza datele pe fire separate; cozile le aduc in bucla principala
        self.camera.listen(self._image_queue.put)
        self.radar.listen(self._radar_queue.put)

        if lead_distance is not None:
            self._spawn_lead(blueprints, spawn_point, lead_distance)

    def _spawn_lead(self, blueprints, spawn_point, distance):
        """Pune o masina la `distance` metri in fata punctului de pornire, pe aceeasi banda.
        O conduce pilotul automat al CARLA, cu viteza pe care i-o dam prin set_lead_speed."""
        start = self._map.get_waypoint(spawn_point.location)
        transform = start.next(distance)[0].transform
        transform.location.z += 0.3  # putin deasupra soselei, ca sa nu se "infiga" in asfalt
        self.lead = self.world.spawn_actor(blueprints.find("vehicle.audi.tt"), transform)
        self._actors.append(self.lead)
        self.lead.set_autopilot(True, TM_PORT)
        self._autopilot_vehicles.append(self.lead)
        self._traffic_manager.auto_lane_change(self.lead, False)
        self.set_lead_speed(0.0)

    def set_lead_speed(self, speed):
        """Viteza pe care masina din fata incearca s-o tina, in m/s."""
        self._traffic_manager.set_desired_speed(self.lead, speed * 3.6)

    def true_lateral_error(self):
        """Adevarul din simulator, pentru validare si metrici (nu pentru reglare):
        cat de departe e centrul masinii de centrul benzii, in metri (pozitiv = la dreapta)."""
        location = self.vehicle.get_transform().location
        waypoint = self._map.get_waypoint(location)  # cel mai apropiat punct de pe centrul benzii
        center = waypoint.transform.location
        right = waypoint.transform.get_right_vector()
        return (location.x - center.x) * right.x + (location.y - center.y) * right.y

    def true_lead_gap(self):
        """Adevarul din simulator, pentru validare si metrici (nu pentru reglare):
        (distanta in metri de la bara noastra din fata la bara din spate a masinii din fata,
         viteza ei minus viteza noastra, in m/s)."""
        centers = self.vehicle.get_transform().location.distance(self.lead.get_transform().location)
        gap = centers - self.vehicle.bounding_box.extent.x - self.lead.bounding_box.extent.x
        lead_velocity = self.lead.get_velocity()
        lead_speed = math.sqrt(lead_velocity.x ** 2 + lead_velocity.y ** 2 + lead_velocity.z ** 2)
        return gap, lead_speed - self.speed()

    def tick(self):
        """Avanseaza simularea cu FIXED_DELTA si intoarce imaginea camerei din acel pas (RGB).
        Punctele radarului din acelasi pas ajung in self.radar_points."""
        frame = self.world.tick()
        self.radar_points = radar_to_points(self._wait_for_frame(self._radar_queue, frame))
        return image_to_rgb(self._wait_for_frame(self._image_queue, frame))

    @staticmethod
    def _wait_for_frame(sensor_queue, frame):
        """Scoate din coada masuratoarea pasului `frame`, sarind peste cele ramase din pasi mai vechi."""
        measurement = sensor_queue.get(timeout=2.0)
        while measurement.frame < frame:
            measurement = sensor_queue.get(timeout=2.0)
        return measurement

    def speed(self):
        """Viteza masinii noastre in m/s."""
        velocity = self.vehicle.get_velocity()
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def apply_control(self, steer_angle, throttle, brake):
        """Trimite comenzile catre masina; se aplica la urmatorul tick().
        steer_angle: unghiul dorit al rotilor in rad (pozitiv = dreapta); throttle, brake: 0..1."""
        # CARLA vrea directia intre -1 si 1, ca fractiune din unghiul maxim al rotilor,
        # pe care apoi o reduce in functie de viteza; impartim la acel factor, ca rotile
        # sa ajunga exact la unghiul cerut
        speed_factor = np.interp(self.speed() * 3.6, self._steer_curve_speeds, self._steer_curve_factors)
        steer = steer_angle / (self.max_steer_angle * speed_factor)
        steer = max(-1.0, min(1.0, steer))
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

    def close(self):
        """Readuce serverul la setarile initiale si sterge masinile si senzorii."""
        # Pilotul automat ruleaza pe alt fir. Ii luam inapoi masinile si lasam simularea sa
        # faca un pas, ca sa nu mai incerce sa conduca o masina pe care tocmai o stergem.
        if self._autopilot_vehicles:
            for vehicle in self._autopilot_vehicles:
                vehicle.set_autopilot(False, TM_PORT)
            self._autopilot_vehicles = []
            self.world.tick()

        # Fara asta serverul ramane in mod sincron si ingheata asteptand tick-uri.
        # Ordinea conteaza: intai iesim din modul sincron, abia apoi stergem actorii.
        self.world.apply_settings(self._original_settings)
        self._traffic_manager.set_synchronous_mode(False)
        for actor in reversed(self._actors):
            if isinstance(actor, carla.Sensor):
                actor.stop()  # senzorul nu mai trimite date; abia apoi il stergem
            actor.destroy()
        self._actors = []

    # Cele doua metode de mai jos fac posibil "with Simulation() as sim:":
    # la iesirea din blocul with, close() se apeleaza garantat, chiar si la eroare.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
