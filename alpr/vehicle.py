"""
Vehicle detection + tracking using Ultralytics YOLO with built-in ByteTrack.

We detect cars/buses/trucks/motorcycles, and the tracker assigns each vehicle a
STABLE id across frames. The plate pipeline then runs *inside* each vehicle box,
so we get exactly one plate per vehicle id.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# COCO class ids for vehicles.
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class VehicleDetection:
    track_id: int
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) full-frame pixels
    cls: str
    conf: float


@dataclass
class VehicleTrack:
    """Aggregated state for one tracked vehicle (one car = one plate).

    Every frame contributes one plate reading. We keep a confidence-weighted
    vote per distinct text so the plate read most *consistently* wins, instead
    of trusting a single (possibly fluke) high-confidence frame. For each text
    we also remember the best crop / box from its strongest frame.
    """
    id: int
    bbox: tuple[int, int, int, int]
    cls: str = "vehicle"
    hits: int = 0
    last_frame_idx: int = 0

    _votes: dict[str, float] = field(default_factory=dict)   # text -> summed conf
    _best_conf: dict[str, float] = field(default_factory=dict)
    _best_crop: dict[str, object] = field(default_factory=dict)
    _best_box: dict[str, object] = field(default_factory=dict)
    _best_sharp: dict[str, float] = field(default_factory=dict)

    def add_reading(self, text: str, conf: float, crop, plate_bbox, sharpness: float = 0.0) -> None:
        if not text:
            return
        # Vote on the text (most consistently read number wins).
        self._votes[text] = self._votes.get(text, 0.0) + conf
        if conf > self._best_conf.get(text, 0.0):
            self._best_conf[text] = conf
        # Keep the SHARPEST crop/box for the snapshot (decoupled from confidence).
        if sharpness >= self._best_sharp.get(text, -1.0):
            self._best_sharp[text] = sharpness
            self._best_box[text] = plate_bbox
            if crop is not None and getattr(crop, "size", 0) > 0:
                self._best_crop[text] = crop

    @property
    def top_text(self) -> str:
        if not self._votes:
            return ""
        return max(self._votes.items(), key=lambda kv: kv[1])[0]

    @property
    def top_conf(self) -> float:
        return self._best_conf.get(self.top_text, 0.0)

    @property
    def top_crop(self):
        return self._best_crop.get(self.top_text)

    @property
    def plate_bbox(self):
        return self._best_box.get(self.top_text)


class VehicleDetector:
    def __init__(
        self,
        model_name: str = "yolov8m.pt",
        conf: float = 0.40,
        device: str = "cpu",
        classes: list[int] | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.conf = conf
        self.classes = classes if classes is not None else list(VEHICLE_CLASSES.keys())
        # Ultralytics device: "cpu" or 0 (first CUDA gpu).
        self.device = 0 if str(device).lower() == "cuda" else "cpu"

    def track(self, frame) -> list[VehicleDetection]:
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            classes=self.classes,
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False,
        )
        out: list[VehicleDetection] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return out

        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for box, tid, c, cf in zip(boxes, ids, clss, confs):
            x1, y1, x2, y2 = (int(v) for v in box)
            out.append(
                VehicleDetection(
                    track_id=int(tid),
                    bbox=(x1, y1, x2, y2),
                    cls=VEHICLE_CLASSES.get(int(c), "vehicle"),
                    conf=float(cf),
                )
            )
        return out
