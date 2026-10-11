"""Regulatoarele masinii: Stanley pentru directie, PI pentru viteza."""

import math


class StanleyController:
    """Regulator lateral Stanley, cu compensarea curburii drumului:
    unghi de volan = eroare de orientare + arctan(k * e / v) + arctan(L * curbura).

    - eroarea de orientare aliniaza masina cu banda (rolul partii derivative);
    - arctan(k * e / v) o trage spre centrul benzii (partea proportionala), cu un
      castig care scade singur cand viteza creste;
    - arctan(L * curbura) este unghiul de care are nevoie curba insasi (feed-forward):
      curbura e o perturbatie pe care o masuram, deci o compensam direct.
    """

    def __init__(self, gain, wheelbase, softening=1.0):
        self.gain = gain            # k: cat de energic corecteaza eroarea laterala (1/s)
        self.wheelbase = wheelbase  # L: ampatamentul masinii (m)
        self.softening = softening  # m/s adaugati la viteza, ca sa nu impartim la ~0 la pornire

    def compute(self, lateral_offset, heading_error, speed, curvature=0.0):
        """lateral_offset: m, pozitiv = masina e la dreapta centrului benzii.
        heading_error: rad, pozitiv = banda o ia la dreapta fata de masina.
        speed: m/s. curvature: 1/m, pozitiva = curba la dreapta.
        Intoarce unghiul dorit al rotilor in rad, pozitiv = spre dreapta."""
        # Minus: daca masina e la dreapta centrului, trebuie sa viram la stanga
        cross_track_term = math.atan2(-self.gain * lateral_offset, speed + self.softening)
        # O masina cu ampatamentul L merge pe un cerc de raza R cu rotile virate la arctan(L / R)
        feed_forward = math.atan(self.wheelbase * curvature)
        return heading_error + cross_track_term + feed_forward


class AccController:
    """Adaptive cruise control: bucla exterioara a unei reglari in cascada.
    Din distanta pana la masina din fata calculeaza ce viteza ar trebui sa avem;
    viteza aceea devine referinta pentru regulatorul de viteza (bucla interioara)."""

    def __init__(self, time_gap, standstill_distance, gain):
        self.time_gap = time_gap                        # intervalul de timp pastrat (s)
        self.standstill_distance = standstill_distance  # distanta pastrata cand stam pe loc (m)
        self.gain = gain    # cati m/s adaugam sau scadem pentru fiecare metru de eroare (1/s)

    def desired_distance(self, speed):
        """Distanta pe care vrem s-o pastram la viteza `speed` (m/s): creste cu viteza."""
        return self.standstill_distance + self.time_gap * speed

    def target_speed(self, distance, relative_speed, speed):
        """distance: m pana la masina din fata. relative_speed: viteza ei minus a noastra (m/s).
        speed: viteza noastra (m/s). Intoarce viteza pe care ar trebui s-o avem (m/s)."""
        lead_speed = speed + relative_speed
        distance_error = distance - self.desired_distance(speed)  # pozitiva = suntem prea departe
        # Mergem cu viteza masinii din fata, corectata proportional cu eroarea de distanta
        return max(0.0, lead_speed + self.gain * distance_error)


class SpeedController:
    """Regulator PI de viteza: din eroarea de viteza scoate acceleratie sau frana."""

    def __init__(self, kp, ki, dt, max_throttle=1.0, max_brake=1.0):
        self.kp = kp
        self.ki = ki
        self.dt = dt                      # perioada de esantionare (s)
        self.max_throttle = max_throttle  # limitele comenzilor, intre 0 si 1
        self.max_brake = max_brake
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
            return min(command, self.max_throttle), 0.0
        return 0.0, min(-command, self.max_brake)
