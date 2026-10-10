"""Regulatoarele masinii: Stanley pentru directie, PI pentru viteza."""

import math


class StanleyController:
    """Regulator lateral Stanley: unghi de volan = eroare de orientare + arctan(k * e / v).

    - eroarea de orientare aliniaza masina cu banda (rolul partii derivative);
    - arctan(k * e / v) o trage spre centrul benzii (partea proportionala), cu un
      castig care scade singur cand viteza creste.
    """

    def __init__(self, gain, softening=1.0):
        self.gain = gain            # k: cat de energic corecteaza eroarea laterala (1/s)
        self.softening = softening  # m/s adaugati la viteza, ca sa nu impartim la ~0 la pornire

    def compute(self, lateral_offset, heading_error, speed):
        """lateral_offset: m, pozitiv = masina e la dreapta centrului benzii.
        heading_error: rad, pozitiv = banda o ia la dreapta fata de masina.
        speed: m/s. Intoarce unghiul dorit al rotilor in rad, pozitiv = spre dreapta."""
        # Minus: daca masina e la dreapta centrului, trebuie sa viram la stanga
        cross_track_term = math.atan2(-self.gain * lateral_offset, speed + self.softening)
        return heading_error + cross_track_term


class SpeedController:
    """Regulator PI de viteza: din eroarea de viteza scoate acceleratie sau frana."""

    def __init__(self, kp, ki, dt):
        self.kp = kp
        self.ki = ki
        self.dt = dt            # perioada de esantionare (s)
        self._integral = 0.0

    def compute(self, target_speed, speed):
        """Viteze in m/s. Intoarce (throttle, brake), fiecare intre 0 si 1."""
        error = target_speed - speed
        self._integral += error * self.dt
        # Anti-windup: integrala nu are voie sa ceara singura mai mult decat comanda maxima
        limit = 1.0 / self.ki
        self._integral = max(-limit, min(limit, self._integral))

        command = self.kp * error + self.ki * self._integral
        if command >= 0:
            return min(command, 1.0), 0.0
        return 0.0, min(-command, 1.0)
