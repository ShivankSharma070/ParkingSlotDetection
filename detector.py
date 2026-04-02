"""
Smart Parking Slot Detection Engine
====================================
Uses a trained YOLOv8s object detection model to classify
parking slots as 'car' (occupied) or 'free' (vacant).

Model: parking-detection-ewm7h v1
Classes: car (class_id=0), free (class_id=1)
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO


class ParkingSlotDetector:
    """
    Parking slot detection engine using a pre-trained
    YOLOv8s object detection model for identifying
    vacant and occupied parking spaces.
    """

    MODEL_TYPE = "YOLOv8s Object Detection"
    CLASSES = {0: "car", 1: "free"}

    # Colors in BGR for OpenCV
    COLORS = {
        "car": {
            "bgr": (71, 71, 235),
            "rgb": (235, 71, 71),
            "hex": "#EB4747",
            "label": "Occupied",
        },
        "free": {
            "bgr": (106, 210, 0),
            "rgb": (0, 210, 106),
            "hex": "#00D26A",
            "label": "Vacant",
        },
    }

    def __init__(self, model_path="model_local/best.pt"):
        self.model = YOLO(model_path)

    def predict(self, image_path, confidence=40, overlap=30):
        """Run local inference on an image file."""
        start = time.time()
        
        results = self.model.predict(
            source=image_path,
            conf=confidence / 100.0,
            iou=overlap / 100.0,
            verbose=False
        )
        elapsed = time.time() - start

        predictions_list = []
        result = results[0]
        
        for box in result.boxes:
            x, y, w, h = box.xywh[0].tolist()
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            
            cls_name = self.CLASSES.get(cls_id, self.model.names.get(cls_id, "unknown"))
            
            predictions_list.append({
                "class": cls_name,
                "confidence": conf_val,
                "x": x, "y": y, "width": w, "height": h
            })

        return {
            "predictions": predictions_list,
            "_inference_time": round(elapsed, 3)
        }

    def annotate(self, image, predictions, show_labels=True):
        """Draw bounding boxes and labels on the image."""
        ann = image.copy()
        h, w = ann.shape[:2]
        font_scale = max(0.35, min(w / 1600, 0.65))
        thickness = max(1, int(w / 800))
        border = max(2, int(w / 500))

        for p in predictions.get("predictions", []):
            pw, ph = int(p["width"]), int(p["height"])
            x1 = max(0, int(p["x"] - pw / 2))
            y1 = max(0, int(p["y"] - ph / 2))
            x2 = min(w, int(p["x"] + pw / 2))
            y2 = min(h, int(p["y"] + ph / 2))

            cls = p["class"]
            conf = p["confidence"]
            color = self.COLORS.get(cls, {"bgr": (255, 255, 255)})["bgr"]

            # Semi-transparent fill
            ov = ann.copy()
            cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(ov, 0.18, ann, 0.82, 0, ann)

            # Border
            cv2.rectangle(ann, (x1, y1), (x2, y2), color, border)

            if show_labels:
                label = f"{cls} {conf:.0%}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th_), _ = cv2.getTextSize(label, font, font_scale, thickness)
                ly1 = max(0, y1 - th_ - 10)
                cv2.rectangle(ann, (x1, ly1), (x1 + tw + 8, y1), color, -1)
                cv2.putText(
                    ann, label, (x1 + 4, y1 - 4),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
                )

        return ann

    @staticmethod
    def get_statistics(predictions):
        """Calculate occupancy statistics."""
        preds = predictions.get("predictions", [])
        total = len(preds)
        occupied = sum(1 for p in preds if p["class"] == "car")
        free = sum(1 for p in preds if p["class"] == "free")
        return {
            "total_slots": total,
            "occupied": occupied,
            "free": free,
            "occupancy_rate": round(occupied / total * 100, 1) if total else 0.0,
        }
