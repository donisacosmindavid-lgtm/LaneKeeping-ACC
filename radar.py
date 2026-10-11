"""Pasul 5: din norul de puncte al radarului alegem masina din fata.

Radarul intoarce la fiecare pas zeci de puncte: unele de pe masina din fata, dar
multe de pe asfalt, parapete si alte benzi. Pastram doar punctele aflate pe banda
noastra, deasupra soselei si care apartin unui obiect in miscare (sau tintei pe
care o urmaream deja), apoi luam cel mai apropiat grup de puncte.
"""

import numpy as np

CORRIDOR_HALF_WIDTH = 1.2    # cat de departe de centrul benzii poate fi un punct (m)
MIN_HEIGHT = -0.5            # inaltimea minima fata de radar (m); mai jos e asfaltul
MAX_HEIGHT = 1.0             # inaltimea maxima fata de radar (m); mai sus sunt poduri, panouri
CLUSTER_DEPTH = 2.0          # punctele aceluiasi obiect stau in cel mult atatia metri in adancime
MIN_CLUSTER_POINTS = 3       # cate puncte trebuie sa aiba un grup ca sa-l credem obiect
MIN_OBJECT_SPEED = 1.0       # sub aceasta viteza fata de sosea, un obiect e considerat fix (m/s)
TRACK_GATE = 3.0             # cat de departe de tinta urmarita cautam punctele ei (m)
MAX_COAST_FRAMES = 10        # cate cadre tinem minte tinta dupa ce nu o mai vedem (0.5 s)


class LeadTracker:
    """Gaseste masina din fata si o urmareste de la un cadru la altul."""

    def __init__(self):
        self._last_distance = None   # distanta la care era tinta ultima data cand am vazut-o
        self._frames_unseen = 0

    def update(self, points, ego_speed, curvature=0.0):
        """points: matrice cu randuri [viteza, azimut, altitudine, distanta]
        (vezi simulation.radar_to_points). ego_speed: viteza noastra in m/s.
        curvature: curbura benzii in 1/m, pozitiva = la dreapta; culoarul de cautare urmeaza banda.
        Intoarce (distanta in m, viteza relativa in m/s, pozitiva = se departeaza) sau None."""
        lead = self._find(points, ego_speed, curvature)
        if lead is not None:
            self._last_distance = lead[0]
            self._frames_unseen = 0
        else:
            self._frames_unseen += 1
            if self._frames_unseen > MAX_COAST_FRAMES:
                self._last_distance = None
        return lead

    def _find(self, points, ego_speed, curvature):
        if len(points) == 0:
            return None
        velocity, azimuth, altitude, depth = points.T

        # Din coordonate polare (distanta + doua unghiuri) in metri fata de radar
        forward = depth * np.cos(altitude) * np.cos(azimuth)
        lateral = depth * np.cos(altitude) * np.sin(azimuth)   # pozitiv = la dreapta
        height = depth * np.sin(altitude)                      # pozitiv = in sus

        # Intr-o curba, centrul benzii se departeaza lateral cu curbura * distanta^2 / 2
        lane_center = curvature * forward ** 2 / 2
        in_lane = np.abs(lateral - lane_center) < CORRIDOR_HALF_WIDTH
        above_road = (height > MIN_HEIGHT) & (height < MAX_HEIGHT)

        # Radarul masoara viteza fata de noi. Un obiect fix "vine" spre noi cu viteza noastra,
        # deci viteza lui fata de sosea este viteza masurata + viteza noastra pe acea directie.
        object_speed = velocity + ego_speed * np.cos(azimuth)
        moving = np.abs(object_speed) > MIN_OBJECT_SPEED
        # Obiectele fixe le ignoram (parapete, stalpi), in afara de tinta pe care o urmaream
        # deja: o masina care a franat pana la oprire trebuie sa ramana tinta.
        if self._last_distance is not None:
            tracked = np.abs(forward - self._last_distance) < TRACK_GATE
        else:
            tracked = np.zeros(len(points), dtype=bool)

        candidates = np.flatnonzero(in_lane & above_road & (moving | tracked))
        if len(candidates) < MIN_CLUSTER_POINTS:
            return None

        # Parcurgem candidatii de la cel mai apropiat si luam primul grup destul de numeros
        candidates = candidates[np.argsort(forward[candidates])]
        distances = forward[candidates]
        for distance in distances:
            group = candidates[(distances >= distance) & (distances <= distance + CLUSTER_DEPTH)]
            if len(group) >= MIN_CLUSTER_POINTS:
                return float(distance), float(np.median(velocity[group]))
        return None
