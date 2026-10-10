"""Legatura cu simulatorul CARLA: harta, modul sincron, masina si camera frontala.

Folosire:
    with Simulation() as sim:
        rgb = sim.tick()   # avanseaza simularea cu un pas si intoarce imaginea camerei
"""

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


def image_to_rgb(image):
    """Transforma imaginea CARLA intr-o matrice numpy (inaltime, latime, 3) in ordine RGB."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # CARLA trimite B, G, R, A
    # Scoatem A, inversam in R, G, B si facem o copie proprie (independenta de memoria CARLA)
    return np.ascontiguousarray(array[:, :, :3][:, :, ::-1])


class Simulation:
    """O masina pe autostrada din Town04, cu camera frontala, in mod sincron."""

    def __init__(self, autopilot=True):
        self.client = carla.Client(HOST, PORT)
        self.client.set_timeout(120.0)  # incarcarea unei harti poate dura mult

        self.world = self.client.get_world()
        if MAP_NAME not in self.world.get_map().name:
            self.world = self.client.load_world(MAP_NAME)

        self._original_settings = self.world.get_settings()
        self._traffic_manager = self.client.get_trafficmanager(TM_PORT)
        self._actors = []
        self._image_queue = queue.Queue()
        self.vehicle = None
        self.camera = None

        try:
            self._setup(autopilot)
        except Exception:
            self.close()  # nu lasam serverul in mod sincron daca pornirea esueaza la jumatate
            raise

    def _setup(self, autopilot):
        # Mod sincron: serverul calculeaza un pas doar cand ii cerem noi, cu world.tick()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA
        self.world.apply_settings(settings)
        self._traffic_manager.set_synchronous_mode(True)

        blueprints = self.world.get_blueprint_library()

        vehicle_bp = blueprints.find("vehicle.tesla.model3")
        vehicle_bp.set_attribute("role_name", "hero")
        spawn_point = self.world.get_map().get_spawn_points()[SPAWN_INDEX]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self._actors.append(self.vehicle)
        if autopilot:
            self.vehicle.set_autopilot(True, TM_PORT)
            # Pilotul automat ramane pe banda lui, ca sa avem un test curat pentru detectie
            self._traffic_manager.auto_lane_change(self.vehicle, False)

        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
        camera_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
        camera_bp.set_attribute("fov", str(CAMERA_FOV))
        self.camera = self.world.spawn_actor(camera_bp, CAMERA_TRANSFORM, attach_to=self.vehicle)
        self._actors.append(self.camera)

        # Camera livreaza imaginile pe un fir separat; coada le aduce in bucla principala
        self.camera.listen(self._image_queue.put)

    def tick(self):
        """Avanseaza simularea cu FIXED_DELTA si intoarce imaginea camerei din acel pas (RGB)."""
        frame = self.world.tick()
        image = self._image_queue.get(timeout=2.0)
        while image.frame < frame:  # sarim peste imagini ramase din pasi mai vechi
            image = self._image_queue.get(timeout=2.0)
        return image_to_rgb(image)

    def close(self):
        """Readuce serverul la setarile initiale si sterge masina si camera."""
        # Fara asta serverul ramane in mod sincron si ingheata asteptand tick-uri
        self.world.apply_settings(self._original_settings)
        self._traffic_manager.set_synchronous_mode(False)
        for actor in reversed(self._actors):
            actor.destroy()
        self._actors = []

    # Cele doua metode de mai jos fac posibil "with Simulation() as sim:":
    # la iesirea din blocul with, close() se apeleaza garantat, chiar si la eroare.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
