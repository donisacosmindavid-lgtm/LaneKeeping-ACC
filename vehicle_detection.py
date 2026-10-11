"""Pasul 6: detectia vehiculelor cu YOLO nano preantrenat.

Spre deosebire de detectia benzii si de radar, aici nu sunt reguli scrise de noi:
YOLO este o retea neuronala antrenata de altii pe imagini etichetate (setul COCO).
Noi doar o folosim (inferenta), fara antrenare. In v1 rezultatul e doar afisat
peste imagine; nu intra in bucla de reglare.
"""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Fisierul modelului; la prima rulare se descarca singur de pe serverele Ultralytics
MODEL_PATH = Path(__file__).parent / "models" / "yolov8n.pt"
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}  # din cele 80 de clase COCO
MIN_CONFIDENCE = 0.25    # sub aceasta incredere, ignoram detectia
DEVICE = 0               # 0 = prima placa video (CUDA); "cpu" = pe procesor


class VehicleDetector:
    def __init__(self):
        self._model = YOLO(str(MODEL_PATH))
        # Prima rulare incarca modelul pe placa video si dureaza mult; o facem aici,
        # pe o imagine goala, ca primul cadru real sa nu astepte
        self.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    def detect(self, rgb):
        """Primeste imaginea camerei (RGB) si intoarce lista vehiculelor gasite, fiecare ca
        (clasa, incredere intre 0 si 1, x_stanga, y_sus, x_dreapta, y_jos), in pixeli."""
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # Ultralytics asteapta ordinea BGR
        result = self._model.predict(bgr, device=DEVICE, conf=MIN_CONFIDENCE, verbose=False)[0]

        vehicles = []
        for class_id, confidence, box in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
            label = self._model.names[int(class_id)]
            if label in VEHICLE_CLASSES:
                x1, y1, x2, y2 = (int(value) for value in box)
                vehicles.append((label, float(confidence), x1, y1, x2, y2))
        return vehicles


def draw_vehicles(image, vehicles):
    """Deseneaza peste imagine cate un dreptunghi cu eticheta pentru fiecare vehicul."""
    for label, confidence, x1, y1, x2, y2 in vehicles:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 160, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 160, 0), 2)
