"""
Simple IOU tracker that stabilises plate readings across frames and finalises
one result per vehicle.

For every tracked plate we remember the SINGLE highest-confidence OCR reading we
ever saw, together with the cropped plate image from that exact moment. When the
vehicle leaves the frame (the track goes stale) the plate is "finalised": we
report that best reading and its crop. This is what the user sees as the final
plate number for each car that passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .pipeline import PlateReading


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    bbox: tuple[int, int, int, int]
    hits: int = 1
    time_since_update: int = 0

    # The single best (most confident) reading ever seen for this plate.
    top_text: str = ""
    top_conf: float = 0.0
    top_crop: object = None          # numpy BGR image of the plate at its best moment
    last_frame_idx: int = 0

    def update(self, reading: PlateReading, crop=None, frame_idx: int = 0) -> None:
        self.bbox = reading.bbox
        self.hits += 1
        self.time_since_update = 0
        self.last_frame_idx = frame_idx
        # Keep only the most confident reading + the crop from that frame.
        if reading.text and reading.ocr_conf > self.top_conf:
            self.top_conf = reading.ocr_conf
            self.top_text = reading.text
            if crop is not None and getattr(crop, "size", 0) > 0:
                self.top_crop = crop.copy()


class PlateTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 15,
        min_hits: int = 3,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: list[Track] = []
        self._next_id = 1
        self._newly_finalized: list[Track] = []

    def _crop_for(self, frame, bbox: tuple[int, int, int, int]):
        if frame is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def update(self, readings: list[PlateReading], frame=None, frame_idx: int = 0) -> list[Track]:
        """Match this frame's readings to existing tracks; return live tracks.

        Stale tracks (vehicles that have left) are moved to the finalized buffer,
        retrievable via `pop_finalized()`.
        """
        for t in self._tracks:
            t.time_since_update += 1

        # Greedy IOU matching, highest overlap first.
        pairs = []
        for ti, track in enumerate(self._tracks):
            for ri, reading in enumerate(readings):
                score = iou(track.bbox, reading.bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, ti, ri))
        pairs.sort(reverse=True)

        used_tracks: set[int] = set()
        used_readings: set[int] = set()
        for _score, ti, ri in pairs:
            if ti in used_tracks or ri in used_readings:
                continue
            crop = self._crop_for(frame, readings[ri].bbox)
            self._tracks[ti].update(readings[ri], crop=crop, frame_idx=frame_idx)
            used_tracks.add(ti)
            used_readings.add(ri)

        # Spawn new tracks for unmatched readings.
        for ri, reading in enumerate(readings):
            if ri in used_readings:
                continue
            track = Track(id=self._next_id, bbox=reading.bbox, last_frame_idx=frame_idx)
            if reading.text:
                track.top_text = reading.text
                track.top_conf = reading.ocr_conf
                track.top_crop = self._crop_for(frame, reading.bbox)
                if track.top_crop is not None:
                    track.top_crop = track.top_crop.copy()
            self._tracks.append(track)
            self._next_id += 1

        # Finalise + drop stale tracks.
        live: list[Track] = []
        for t in self._tracks:
            if t.time_since_update > self.max_age:
                if t.hits >= self.min_hits and t.top_text:
                    self._newly_finalized.append(t)
            else:
                live.append(t)
        self._tracks = live
        return self._tracks

    def live_tracks(self) -> list[Track]:
        """Tracks visible right now and confirmed (for live overlay)."""
        return [
            t
            for t in self._tracks
            if t.hits >= self.min_hits and t.time_since_update == 0 and t.top_text
        ]

    def pop_finalized(self) -> list[Track]:
        out = self._newly_finalized
        self._newly_finalized = []
        return out

    def flush(self) -> list[Track]:
        """Finalise every remaining track (call at end of stream)."""
        for t in self._tracks:
            if t.hits >= self.min_hits and t.top_text:
                self._newly_finalized.append(t)
        self._tracks = []
        return self.pop_finalized()
