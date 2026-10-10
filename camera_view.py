"""Pasul 2: masina pe Town04 cu o camera frontala, imaginea afisata intr-o fereastra.

Rulare (cu serverul CARLA pornit):
    venv\\Scripts\\python.exe camera_view.py
Iesire: ESC sau inchiderea ferestrei.
"""

import queue

import carla
import numpy as np
import pygame

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
    return array[:, :, :3][:, :, ::-1]                     # scoatem A si inversam in R, G, B


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(120.0)  # incarcarea unei harti poate dura mult

    world = client.get_world()
    if MAP_NAME not in world.get_map().name:
        world = client.load_world(MAP_NAME)

    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(TM_PORT)
    actors = []

    pygame.init()
    display = pygame.display.set_mode((IMAGE_WIDTH, IMAGE_HEIGHT))
    pygame.display.set_caption("ADAS - camera frontala")
    clock = pygame.time.Clock()

    try:
        # Mod sincron: serverul calculeaza un pas doar cand ii cerem noi, cu world.tick()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)

        blueprints = world.get_blueprint_library()

        vehicle_bp = blueprints.find("vehicle.tesla.model3")
        vehicle_bp.set_attribute("role_name", "hero")
        spawn_point = world.get_map().get_spawn_points()[SPAWN_INDEX]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actors.append(vehicle)
        vehicle.set_autopilot(True, TM_PORT)

        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
        camera_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
        camera_bp.set_attribute("fov", str(CAMERA_FOV))
        camera = world.spawn_actor(camera_bp, CAMERA_TRANSFORM, attach_to=vehicle)
        actors.append(camera)

        # Camera livreaza imaginile pe un fir separat; coada le aduce in bucla principala
        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            world.tick()                           # simularea avanseaza cu FIXED_DELTA
            image = image_queue.get(timeout=2.0)   # imaginea produsa in acest pas

            rgb = image_to_rgb(image)
            # pygame vrea (latime, inaltime, 3), numpy are (inaltime, latime, 3)
            surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            display.blit(surface, (0, 0))
            pygame.display.flip()

            clock.tick(1.0 / FIXED_DELTA)          # cel mult 20 de cadre pe secunda reala
    finally:
        # Fara asta serverul ramane in mod sincron si ingheata asteptand tick-uri
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)
        for actor in reversed(actors):
            actor.destroy()
        pygame.quit()


if __name__ == "__main__":
    main()
