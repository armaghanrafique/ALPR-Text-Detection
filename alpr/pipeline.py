"""
Thin wrapper around fast-alpr that returns clean, structured plate readings.

fast-alpr bundles:
  * a YOLOv9 ONNX license-plate detector (open-image-models)
  * a MobileViT ONNX plate-OCR model (fast-plate-ocr)

Both run on CPU via onnxruntime, and on GPU automatically once
`onnxruntime-gpu` is installed (no code change required).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlateReading:
    """One plate detected in one frame."""
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixels
    det_conf: float                   # detector confidence 0..1
    text: str                         # OCR text ("" if OCR failed)
    ocr_conf: float                   # OCR confidence 0..1


def _to_conf(value) -> float:
    """Normalise a confidence into a single float.

    fast-plate-ocr returns per-character confidences as a list; the detector
    returns a plain float. Average a list so we get one stable score.
    """
    if value is None:
        return 0.0
    if isinstance(value, (list, tuple)):
        vals = [float(v) for v in value if v is not None]
        return sum(vals) / len(vals) if vals else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_bbox(detection) -> tuple[int, int, int, int]:
    """Pull (x1, y1, x2, y2) out of a fast-alpr / open-image-models box."""
    box = getattr(detection, "bounding_box", detection)
    # Most builds expose .x1/.y1/.x2/.y2 attributes.
    if all(hasattr(box, a) for a in ("x1", "y1", "x2", "y2")):
        return int(box.x1), int(box.y1), int(box.x2), int(box.y2)
    # Fallbacks for tuple / list style boxes.
    if isinstance(box, (tuple, list)) and len(box) >= 4:
        return tuple(int(v) for v in box[:4])  # type: ignore[return-value]
    raise AttributeError(f"Cannot read bounding box from {detection!r}")


class PlatePipeline:
    """Loads the models once and turns frames into PlateReading lists."""

    def __init__(
        self,
        detector_model: str = "yolo-v9-s-608-license-plate-end2end",
        ocr_model: str = "global-plates-mobile-vit-v2-model",
        device: str = "cpu",
        min_detection_conf: float = 0.40,
        min_ocr_conf: float = 0.50,
    ) -> None:
        # Imported here so `python download_models.py` can give a clean error
        # message if dependencies are not installed yet.
        from fast_alpr import ALPR

        self.min_detection_conf = min_detection_conf
        self.min_ocr_conf = min_ocr_conf
        self.device = device

        self._alpr = ALPR(detector_model=detector_model, ocr_model=ocr_model)

    def predict(self, frame) -> list[PlateReading]:
        """Run detection + OCR on a single BGR frame (numpy array)."""
        raw_results = self._alpr.predict(frame)
        readings: list[PlateReading] = []

        for res in raw_results:
            detection = getattr(res, "detection", None)
            if detection is None:
                continue

            det_conf = _to_conf(getattr(detection, "confidence", 0.0))
            if det_conf < self.min_detection_conf:
                continue

            try:
                bbox = _extract_bbox(detection)
            except AttributeError:
                continue

            ocr = getattr(res, "ocr", None)
            text = ""
            ocr_conf = 0.0
            if ocr is not None:
                text = (getattr(ocr, "text", "") or "").strip()
                ocr_conf = _to_conf(getattr(ocr, "confidence", 0.0))

            if text and ocr_conf < self.min_ocr_conf:
                # Keep the box but drop a low-confidence reading.
                text, ocr_conf = "", 0.0

            readings.append(
                PlateReading(bbox=bbox, det_conf=det_conf, text=text, ocr_conf=ocr_conf)
            )

        return readings
