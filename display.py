"""Fereastra in care afisam imagini (pygame)."""

import pygame


class Window:
    def __init__(self, width, height, title, fps):
        pygame.init()
        self._display = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self._clock = pygame.time.Clock()
        self._fps = fps

    def show(self, rgb):
        """Afiseaza o imagine RGB (inaltime, latime, 3).
        Intoarce False daca utilizatorul a cerut iesirea (ESC sau inchiderea ferestrei)."""
        keep_running = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keep_running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                keep_running = False

        # pygame vrea (latime, inaltime, 3), numpy are (inaltime, latime, 3)
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        self._display.blit(surface, (0, 0))
        pygame.display.flip()

        self._clock.tick(self._fps)  # cel mult `fps` cadre pe secunda reala
        return keep_running

    def close(self):
        pygame.quit()
