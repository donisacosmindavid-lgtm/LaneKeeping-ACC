"""Pasul 7: metricile unei rulari, scrise intr-un fisier CSV (un rand pe cadru).

Scriere (din lane_keeping.py):
    venv\\Scripts\\python.exe lane_keeping.py --log results\\test.csv --steps 1200
Rezumatul unui fisier:
    venv\\Scripts\\python.exe metrics.py results\\test.csv
"""

import csv
import sys
from pathlib import Path

import numpy as np

# Coloanele fisierului. Cele cu "true" sunt adevarul din simulator, pentru comparatie;
# restul sunt ce au masurat senzorii nostri si ce a comandat regulatorul.
COLUMNS = [
    "step",                     # numarul pasului de simulare
    "time_s",                   # timpul de simulare (s)
    "speed_kmh",                # viteza noastra
    "target_speed_kmh",         # viteza ceruta regulatorului de viteza
    "lane_detected",            # 1 daca am detectat banda in acest cadru
    "lateral_error_m",          # eroarea laterala din camera, la 5 m in fata camerei
    "lateral_error_true_m",     # eroarea laterala reala, sub masina
    "steer_deg",                # unghiul de volan comandat
    "lead_detected",            # 1 daca radarul vede o masina in fata
    "lead_distance_m",          # distanta din radar
    "lead_distance_true_m",     # distanta reala
    "lead_rel_speed_ms",        # viteza relativa din radar (pozitiva = se departeaza)
    "lead_rel_speed_true_ms",   # viteza relativa reala
    "desired_distance_m",       # distanta pe care ACC incearca s-o pastreze
    "vehicles_seen",            # cate vehicule a gasit YOLO in imagine
    "latency_ms",               # cat a durat prelucrarea cadrului, de la imagine la comenzi
]
LATENCY_BUDGET_MS = 50.0        # pasul de simulare: un cadru trebuie prelucrat in acest timp


class MetricsLogger:
    """Scrie cate un rand pe cadru. Folosire: with MetricsLogger(cale) as logger: logger.log(...)"""

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=COLUMNS)
        self._writer.writeheader()

    def log(self, **values):
        """Primeste valorile ca argumente cu nume (coloana=valoare). O valoare None
        (de exemplu distanta cand nu e nicio masina in fata) ramane celula goala."""
        row = {}
        for column in COLUMNS:
            value = values.get(column)
            if value is None:
                row[column] = ""
            elif isinstance(value, float):
                row[column] = f"{value:.4f}"
            else:
                row[column] = value
        self._writer.writerow(row)

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def load(path):
    """Citeste un fisier de metrici. Intoarce un dictionar coloana -> matrice numpy;
    celulele goale devin NaN ("not a number"), pe care functiile nan* din numpy le sar."""
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    return {column: np.array([float(row[column]) if row[column] != "" else np.nan for row in rows])
            for column in COLUMNS}


def summarize(path):
    """Calculeaza cifrele principale ale unei rulari. Intoarce un dictionar nume -> valoare."""
    data = load(path)
    lateral = data["lateral_error_true_m"]
    latency = data["latency_ms"]
    following = data["lead_detected"] == 1
    distance_error = data["lead_distance_true_m"] - data["desired_distance_m"]
    radar_error = data["lead_distance_m"] - data["lead_distance_true_m"]

    summary = {
        "cadre": len(lateral),
        "durata (s)": data["time_s"][-1],
        "viteza medie (km/h)": data["speed_kmh"].mean(),
        "banda detectata (% cadre)": 100 * data["lane_detected"].mean(),
        "eroare laterala reala, RMS (m)": np.sqrt(np.mean(lateral ** 2)),
        "eroare laterala reala, maxim (m)": np.abs(lateral).max(),
        "latenta, mediana (ms)": np.median(latency),
        "latenta, p95 (ms)": np.percentile(latency, 95),
        "latenta, maxim (ms)": latency.max(),
        f"cadre peste {LATENCY_BUDGET_MS:.0f} ms": int((latency > LATENCY_BUDGET_MS).sum()),
        "masina in fata vazuta de radar (% cadre)": 100 * following.mean(),
    }
    if following.any():
        summary.update({
            "distanta reala minima (m)": np.nanmin(data["lead_distance_true_m"]),
            "eroare de distanta fata de cea dorita, medie (m)": np.nanmean(distance_error[following]),
            "eroare de distanta fata de cea dorita, RMS (m)": np.sqrt(np.nanmean(distance_error[following] ** 2)),
            "abaterea radarului fata de distanta reala, RMS (m)": np.sqrt(np.nanmean(radar_error[following] ** 2)),
        })
    return summary


def main():
    for path in sys.argv[1:]:
        print(path)
        for name, value in summarize(path).items():
            text = f"{value:.2f}" if isinstance(value, float) else str(value)
            print(f"  {name:52s} {text}")


if __name__ == "__main__":
    main()
