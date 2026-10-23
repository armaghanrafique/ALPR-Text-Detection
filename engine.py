"""
Background processing engine used by the web app.

Pipeline (vehicle-first):
  1. Detect + track each VEHICLE (Ultralytics YOLO + ByteTrack) -> stable car id.
  2. Run the plate detector + OCR INSIDE each vehicle box.
  3. Keep the single most-confident plate reading per vehicle id.
  4. When the vehicle leaves the frame, finalise ONE plate for that car -- but
     only if its confidence clears the display threshold (default 90%).

This guarantees one plate per car and one id per car, and avoids low-confidence
junk being shown.
"""

from __future__ import annotations

import os
import threading
import time

import cv2

from alpr import PlateMatcher, PlatePipeline, PlateReading
from alpr.matcher import normalize
from alpr.vehicle import VehicleDetector, VehicleTrack

try:
    from rapidfuzz import fuzz as _fuzz
except Exception:  # pragma: no cover
    _fuzz = None


def _similarity(a: str, b: str) -> float:
    if a == b:
        return 100.0
    if _fuzz is not None:
        return float(_fuzz.ratio(a, b))
    import difflib

    return difflib.SequenceMatcher(None, a, b).ratio() * 100.0


GREEN = (0, 200, 0)
RED = (0, 0, 255)
BLUE = (255, 170, 0)
YELLOW = (0, 215, 255)
ORANGE = (0, 140, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

PLATES_DIR = os.path.join("static", "plates")


def _sharpness(img) -> float:
    """Focus measure: variance of the Laplacian. Higher = sharper."""
    if img is None or getattr(img, "size", 0) == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _center_in(box, roi) -> bool:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def _clip_box(bbox, w, h):
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    return x1, y1, x2, y2


def _draw_label(frame, bbox, line1, line2, color):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    lines = [t for t in (line1, line2) if t]
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick, pad = 0.7, 2, 4
    sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t in lines]
    box_w = max(w for w, _ in sizes) + 2 * pad
    line_h = max(h for _, h in sizes) + pad
    box_h = line_h * len(lines) + pad
    top = max(0, y1 - box_h)
    cv2.rectangle(frame, (x1, top), (x1 + box_w, top + box_h), color, -1)
    y = top + line_h
    for t in lines:
        cv2.putText(frame, t, (x1 + pad, y), font, scale, BLACK, thick, cv2.LINE_AA)
        y += line_h


class Engine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()

        self.running = False
        self.status = "idle"
        self.frame_idx = 0
        self.fps = 0.0
        self._latest_jpeg: bytes | None = None
        self.plates: list[dict] = []          # newest first
        self._plate_seq = 0
        self._seen: dict[str, dict] = {}      # normalised text -> record (dedupe)

        dcfg = cfg.get("dedupe", {})
        self._dedupe_enabled = bool(dcfg.get("enabled", True))
        self._dedupe_similarity = float(dcfg.get("similarity", 90))

        # Only finalise/display plates at or above this confidence (percent).
        self._display_min_conf = float(cfg.get("display_min_confidence", 90))

        # Detection zone + quality gates (read plates only where they're readable).
        rcfg = cfg.get("roi", {})
        self._roi_enabled = bool(rcfg.get("enabled", False))
        self._roi_frac = (
            float(rcfg.get("x1", 0.0)),
            float(rcfg.get("y1", 0.0)),
            float(rcfg.get("x2", 1.0)),
            float(rcfg.get("y2", 1.0)),
        )
        self._min_sharpness = float(cfg.get("min_sharpness", 0.0))
        self._min_plate_width = int(cfg.get("min_plate_width", 0))

        # Built lazily on first start.
        self._pipeline: PlatePipeline | None = None
        self._matcher: PlateMatcher | None = None
        self._vehicle: VehicleDetector | None = None

        os.makedirs(PLATES_DIR, exist_ok=True)

    # ---- model loading ----------------------------------------------------
    def _ensure_models(self):
        if self._pipeline is not None:
            return
        cfg = self.cfg
        self.status = "loading models (first run downloads weights)..."

        self._vehicle = VehicleDetector(
            model_name=cfg.get("vehicle_model", "yolov8m.pt"),
            conf=float(cfg.get("vehicle_conf", 0.40)),
            device=cfg.get("device", "cpu"),
        )
        self._pipeline = PlatePipeline(
            detector_model=cfg.get("detector_model", "yolo-v9-s-608-license-plate-end2end"),
            ocr_model=cfg.get("ocr_model", "global-plates-mobile-vit-v2-model"),
            device=cfg.get("device", "cpu"),
            min_detection_conf=float(cfg.get("min_detection_conf", 0.40)),
            min_ocr_conf=float(cfg.get("min_ocr_conf", 0.40)),
        )
        mcfg = cfg.get("matcher", {})
        if mcfg.get("enabled", False):
            self._matcher = PlateMatcher(
                watchlist_path=mcfg.get("watchlist", "watchlist.txt"),
                fuzzy_threshold=float(mcfg.get("fuzzy_threshold", 85)),
            )

    # ---- public control ---------------------------------------------------
    def start(self, source: str) -> str:
        with self._lock:
            if self.running:
                return "already running"
            self._stop_flag.clear()
            self.plates = []
            self._plate_seq = 0
            self._seen = {}
            for f in os.listdir(PLATES_DIR):
                if f.lower().endswith(".jpg"):
                    try:
                        os.remove(os.path.join(PLATES_DIR, f))
                    except OSError:
                        pass
            self.running = True
            self.status = "starting..."
        self._thread = threading.Thread(target=self._run, args=(source,), daemon=True)
        self._thread.start()
        return "started"

    def stop(self) -> str:
        self._stop_flag.set()
        return "stopping"

    def get_state(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "status": self.status,
                "frame": self.frame_idx,
                "fps": round(self.fps, 1),
                "count": len(self.plates),
                "threshold": self._display_min_conf,
                "settings": self._settings_dict(),
                "plates": list(self.plates),
            }

    def _settings_dict(self) -> dict:
        return {
            "display_min_confidence": self._display_min_conf,
            "roi_enabled": self._roi_enabled,
            "roi": list(self._roi_frac),
            "min_sharpness": self._min_sharpness,
            "min_plate_width": self._min_plate_width,
            "dedupe_similarity": self._dedupe_similarity,
        }

    def update_settings(self, s: dict) -> dict:
        """Apply tuning changes live (no restart needed)."""
        with self._lock:
            if "display_min_confidence" in s:
                self._display_min_conf = float(s["display_min_confidence"])
            if "roi_enabled" in s:
                self._roi_enabled = bool(s["roi_enabled"])
            if "roi" in s and isinstance(s["roi"], (list, tuple)) and len(s["roi"]) == 4:
                self._roi_frac = tuple(float(v) for v in s["roi"])
            if "min_sharpness" in s:
                self._min_sharpness = float(s["min_sharpness"])
            if "min_plate_width" in s:
                self._min_plate_width = int(s["min_plate_width"])
            if "dedupe_similarity" in s:
                self._dedupe_similarity = float(s["dedupe_similarity"])
            return self._settings_dict()

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    # ---- finalisation + dedupe -------------------------------------------
    def _find_duplicate(self, norm: str):
        if not self._dedupe_enabled or not norm:
            return None
        if norm in self._seen:
            return self._seen[norm]
        best, best_score = None, 0.0
        for key, rec in self._seen.items():
            s = _similarity(norm, key)
            if s > best_score:
                best, best_score = rec, s
        if best is not None and best_score >= self._dedupe_similarity:
            return best
        return None

    def _finalize(self, vt: VehicleTrack):
        text = vt.top_text
        conf_pct = round(float(vt.top_conf) * 100, 1)
        # Enforce the display threshold: only confident plates become results.
        if not text or conf_pct < self._display_min_conf:
            return
        norm = normalize(text)

        matched, matched_plate, exact, score = False, None, False, 0.0
        if self._matcher is not None:
            m = self._matcher.match(text)
            matched, matched_plate, exact, score = m.matched, m.plate, m.exact, m.score

        with self._lock:
            dup = self._find_duplicate(norm)
            if dup is not None:
                dup["count"] = dup.get("count", 1) + 1
                dup["last_frame"] = vt.last_frame_idx
                if conf_pct > dup["confidence"]:
                    dup["text"] = text
                    dup["confidence"] = conf_pct
                    dup["matched"] = matched
                    dup["matched_plate"] = matched_plate
                    dup["exact"] = exact
                    dup["match_score"] = round(float(score), 1)
                    fname = f"plate_{dup['seq']:04d}.jpg"
                    if self._save_crop(vt.top_crop, os.path.join(PLATES_DIR, fname)):
                        dup["image"] = f"/static/plates/{fname}"
                        dup["ver"] = dup.get("ver", 0) + 1  # cache-bust on update
                if dup in self.plates:
                    self.plates.remove(dup)
                self.plates.insert(0, dup)
                return

            self._plate_seq += 1
            seq = self._plate_seq
            img_url = ""
            fname = f"plate_{seq:04d}.jpg"
            if self._save_crop(vt.top_crop, os.path.join(PLATES_DIR, fname)):
                img_url = f"/static/plates/{fname}"

            record = {
                "seq": seq,
                "track_id": vt.id,
                "vehicle": vt.cls,
                "text": text,
                "confidence": conf_pct,
                "image": img_url,
                "matched": matched,
                "matched_plate": matched_plate,
                "exact": exact,
                "match_score": round(float(score), 1),
                "frame": vt.last_frame_idx,
                "last_frame": vt.last_frame_idx,
                "count": 1,
                "ver": 0,
            }
            self.plates.insert(0, record)
            if norm:
                self._seen[norm] = record

    def _save_crop(self, crop, path) -> bool:
        """Save the plate crop, upscaling small ones so they're readable."""
        if crop is None or getattr(crop, "size", 0) == 0:
            return False
        img = crop
        h, w = img.shape[:2]
        target_h = 110
        if 0 < h < target_h:
            scale = target_h / h
            img = cv2.resize(
                img, (max(1, int(w * scale)), target_h), interpolation=cv2.INTER_CUBIC
            )
        return bool(cv2.imwrite(path, img))

    def _publish(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self._latest_jpeg = buf.tobytes()

    # ---- worker -----------------------------------------------------------
    def _best_plate_in(self, crop):
        """Highest-confidence plate reading inside a vehicle crop.

        Small / distant vehicle crops are upscaled before OCR so the plate is
        large enough to read; the returned box is mapped back to crop pixels.
        """
        img = crop
        scale = 1.0
        h, w = img.shape[:2]
        m = max(h, w)
        if 0 < m < 640:
            scale = 640.0 / m
            img = cv2.resize(
                img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
            )

        readings = self._pipeline.predict(img)  # type: ignore[union-attr]
        best = None
        for r in readings:
            if r.text and (best is None or r.ocr_conf > best.ocr_conf):
                best = r
        if best is not None and scale != 1.0:
            x1, y1, x2, y2 = best.bbox
            best = PlateReading(
                bbox=(int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)),
                det_conf=best.det_conf,
                text=best.text,
                ocr_conf=best.ocr_conf,
            )
        return best

    def _run(self, source: str):
        try:
            self._ensure_models()
        except Exception as e:
            with self._lock:
                self.status = f"error loading models: {e}"
                self.running = False
            return

        cap_src: object = source
        if isinstance(source, str) and source.isdigit():
            cap_src = int(source)
        cap = cv2.VideoCapture(cap_src)
        if not cap.isOpened():
            with self._lock:
                self.status = f"could not open input: {source}"
                self.running = False
            return

        max_age = int(self.cfg.get("tracker", {}).get("max_age", 15))
        vehicles: dict[int, VehicleTrack] = {}

        with self._lock:
            self.status = "processing"
            self.frame_idx = 0
        t0 = time.time()
        idx = 0
        try:
            while not self._stop_flag.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                h, w = frame.shape[:2]

                # Detection zone in pixels for this frame.
                rx1, ry1, rx2, ry2 = self._roi_frac
                roi = (int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h))

                detections = self._vehicle.track(frame)  # type: ignore[union-attr]

                for d in detections:
                    vt = vehicles.get(d.track_id)
                    if vt is None:
                        vt = VehicleTrack(id=d.track_id, bbox=d.bbox, cls=d.cls)
                        vehicles[d.track_id] = vt
                    vt.bbox = d.bbox
                    vt.cls = d.cls
                    vt.hits += 1
                    vt.last_frame_idx = idx

                    vx1, vy1, vx2, vy2 = _clip_box(d.bbox, w, h)
                    if vx2 <= vx1 or vy2 <= vy1:
                        continue
                    vcrop = frame[vy1:vy2, vx1:vx2]
                    if vcrop.size == 0:
                        continue

                    best = self._best_plate_in(vcrop)
                    if best is not None:
                        px1, py1, px2, py2 = best.bbox
                        full_box = (vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2)

                        # Gate 1: only read plates inside the detection zone.
                        if self._roi_enabled and not _center_in(full_box, roi):
                            continue
                        # Gate 2: plate must be large enough.
                        if (px2 - px1) < self._min_plate_width:
                            continue

                        # pad the crop a little so whole plate is captured
                        pw, ph = px2 - px1, py2 - py1
                        padx, pady = int(pw * 0.06) + 2, int(ph * 0.18) + 2
                        cx1, cy1 = max(0, px1 - padx), max(0, py1 - pady)
                        cx2 = min(vcrop.shape[1], px2 + padx)
                        cy2 = min(vcrop.shape[0], py2 + pady)
                        plate_crop = None
                        if cx2 > cx1 and cy2 > cy1:
                            plate_crop = vcrop[cy1:cy2, cx1:cx2].copy()

                        # Gate 3: reject blurry crops (and keep sharpest as snapshot).
                        sharp = _sharpness(plate_crop)
                        if sharp < self._min_sharpness:
                            continue

                        vt.add_reading(best.text, best.ocr_conf, plate_crop, full_box, sharp)

                # --- draw overlay ---
                if self._roi_enabled:
                    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), YELLOW, 2)
                    cv2.putText(
                        frame, "DETECTION ZONE", (roi[0] + 6, max(14, roi[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2, cv2.LINE_AA,
                    )

                for d in detections:
                    vt = vehicles.get(d.track_id)
                    vx1, vy1, vx2, vy2 = _clip_box(d.bbox, w, h)
                    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), BLUE, 2)
                    cv2.putText(
                        frame, f"{d.cls} #{d.track_id}", (vx1, max(0, vy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2, cv2.LINE_AA,
                    )
                    # Show the plate as soon as it's read. Green = will be saved
                    # (>= threshold); orange = detected but still below threshold.
                    if vt and vt.top_text:
                        conf_pct = vt.top_conf * 100
                        box = vt.plate_bbox or (vx1, vy1, vx2, vy2)
                        if conf_pct >= self._display_min_conf:
                            color = GREEN
                            line2 = f"{conf_pct:.0f}%"
                            if self._matcher is not None and self._matcher.match(vt.top_text).matched:
                                color = RED
                                line2 = "WATCHLIST"
                        else:
                            color = ORANGE
                            line2 = f"{conf_pct:.0f}% (low)"
                        _draw_label(frame, box, vt.top_text, line2, color)

                # --- finalise vehicles that have left ---
                for tid in list(vehicles.keys()):
                    vt = vehicles[tid]
                    if idx - vt.last_frame_idx > max_age:
                        self._finalize(vt)
                        del vehicles[tid]

                elapsed = time.time() - t0
                fps = idx / elapsed if elapsed > 0 else 0.0
                cv2.putText(
                    frame, f"frame {idx} | {fps:.1f} fps | conf>={self._display_min_conf:.0f}%",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA,
                )
                self._publish(frame)
                with self._lock:
                    self.frame_idx = idx
                    self.fps = fps

            # End of stream / stopped: finalise everything still tracked.
            for vt in vehicles.values():
                self._finalize(vt)
        finally:
            cap.release()
            with self._lock:
                self.running = False
                self.status = "stopped" if self._stop_flag.is_set() else "finished"
