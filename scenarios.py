"""Pasul 8: scenariile de test si tabelul cu rezultate.

Fiecare scenariu ruleaza la fel de fiecare data, pe autostrada de dinaintea
intersectiei, si isi scrie metricile in results/<nume>.csv. La final, tabelul
din results/summary.md se genereaza din aceste fisiere.

Rulare (cu serverul CARLA pornit):
    venv\\Scripts\\python.exe scenarios.py             toate scenariile
    venv\\Scripts\\python.exe scenarios.py curba       doar cele numite
    venv\\Scripts\\python.exe scenarios.py --table     doar reface tabelul din fisierele existente
"""

import sys
from pathlib import Path

from lane_keeping import run
from metrics import summarize

RESULTS_DIR = Path(__file__).parent / "results"
ROUTE_LENGTH = 1100.0   # metri; autostrada se termina la ~1200 m intr-o intersectie


def follow_profile(elapsed):
    """Masina din fata isi schimba viteza: 40 km/h, apoi 25, apoi 45."""
    if elapsed < 25:
        return 40 / 3.6
    if elapsed < 45:
        return 25 / 3.6
    return 45 / 3.6


def hard_brake_profile(elapsed):
    """Masina din fata merge cu 45 km/h, se opreste brusc, apoi porneste din nou."""
    if elapsed < 30:
        return 45 / 3.6
    if elapsed < 45:
        return 0.0
    return 40 / 3.6


# nume -> (descriere, vreme, profilul masinii din fata, distanta maxima, numar maxim de pasi)
SCENARIOS = {
    "curba": ("Curbe (147 m si 54 m raza), fara masina in fata", None, None, ROUTE_LENGTH, 2400),
    "franare_brusca": ("Masina din fata se opreste brusc si reporneste", None, hard_brake_profile, None, 1400),
    "urmarire_zi": ("Urmarire cu viteza variabila, zi", None, follow_profile, ROUTE_LENGTH, 2400),
    "urmarire_ploaie_usoara": ("Urmarire cu viteza variabila, ploaie usoara", "soft_rain", follow_profile, ROUTE_LENGTH, 2400),
    "urmarire_ploaie_medie": ("Urmarire cu viteza variabila, ploaie medie", "mid_rain", follow_profile, ROUTE_LENGTH, 2400),
    "urmarire_ploaie_puternica": ("Urmarire cu viteza variabila, ploaie puternica", "hard_rain", follow_profile, ROUTE_LENGTH, 1200),
    "urmarire_noapte": ("Urmarire cu viteza variabila, noapte", "night", follow_profile, ROUTE_LENGTH, 2400),
}

# Coloanele tabelului: (titlu, cheia din metrics.summarize, numar de zecimale)
TABLE_COLUMNS = [
    ("Distanta parcursa (m)", "distanta parcursa (m)", 0),
    ("Banda detectata (%)", "banda detectata (% cadre)", 1),
    ("Eroare laterala RMS (m)", "eroare laterala reala, RMS (m)", 2),
    ("Eroare laterala max (m)", "eroare laterala reala, maxim (m)", 2),
    ("Distanta minima (m)", "distanta reala minima (m)", 1),
    ("Eroare de distanta RMS (m)", "eroare de distanta fata de cea dorita, RMS (m)", 1),
    ("Latenta mediana (ms)", "latenta, mediana (ms)", 0),
    ("Latenta p95 (ms)", "latenta, p95 (ms)", 0),
]


def run_scenario(name):
    description, weather, lead_speed, max_distance, max_steps = SCENARIOS[name]
    print(f"Rulez scenariul {name}: {description}")
    run(lead_speed=lead_speed, weather=weather, log_path=RESULTS_DIR / f"{name}.csv",
        max_steps=max_steps, max_distance=max_distance)


def build_table():
    """Construieste tabelul (Markdown) din fisierele CSV existente si il scrie in summary.md."""
    lines = ["| Scenariu | " + " | ".join(title for title, _, _ in TABLE_COLUMNS) + " |",
             "|---|" + "---|" * len(TABLE_COLUMNS)]
    for name, (description, *_) in SCENARIOS.items():
        path = RESULTS_DIR / f"{name}.csv"
        if not path.exists():
            continue
        summary = summarize(path)
        cells = []
        for _, key, decimals in TABLE_COLUMNS:
            # Scenariile fara masina in fata nu au distanta: celula ramane "-"
            cells.append(f"{summary[key]:.{decimals}f}" if key in summary else "-")
        lines.append(f"| {description} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    (RESULTS_DIR / "summary.md").write_text(table + "\n", encoding="utf-8")
    return table


def main():
    arguments = sys.argv[1:]
    if arguments != ["--table"]:
        for name in arguments or SCENARIOS:
            run_scenario(name)
    print(build_table())


if __name__ == "__main__":
    main()
