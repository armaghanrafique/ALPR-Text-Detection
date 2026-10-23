#!/usr/bin/env python
"""
License Plate Recognition runner.

Reads a video (or webcam), detects + reads number plates, draws each plate's
number on screen, optionally matches against a watchlist of known plates, and
saves an annotated video + a CSV log.

Usage:
    python run.py                          # use settings from config.yaml
    python run.py --input clip.mp4         # process a specific video
    python run.py --input 0                # use the default webcam
    python run.py --no-show                # headless (no preview window)
    python run.py --device cuda            # after installing onnxruntime-gpu
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import cv2
import yaml

from alpr import PlateMatcher, PlatePipeline, PlateTracker


# --- colours (BGR) ---------------------------------------------------------
GREEN = (0, 200, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[warn] config '{path}' not found, using built-in defaults.")
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="License plate recognition")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--input", help="video path or '0' for webcam")
    p.add_argument("--output", help="annotated video output path")
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.add_argument("--no-show", action="store_true", help="disable preview window")
    p.add_argument("--no-save", action="store_true", help="do not write output video")
    return p.parse_args()


def draw_label(frame, bbox, line1: str, line2: str | None, color) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    lines = [t for t in (line1, line2) if t]
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thick = 2
    pad = 4
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


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides config.
    input_src = args.input if args.input is not None else str(cfg.get("input", "0"))
    output_path = args.output or cfg.get("output", "output/annotated.mp4")
    device = args.device or cfg.get("device", "cpu")
    show_window = (not args.no_show) and cfg.get("show_window", True)
    save_video = (not args.no_save) and cfg.get("save_video", True)
    save_csv = cfg.get("save_csv", None)

    # Resolve webcam index ("0" -> 0).
    cap_src: object = input_src
    if isinstance(input_src, str) and input_src.isdigit():
        cap_src = int(input_src)

    print("[info] loading models (first run downloads weights, please wait)...")
    pipeline = PlatePipeline(
        detector_model=cfg.get("detector_model", "yolo-v9-s-608-license-plate-end2end"),
        ocr_model=cfg.get("ocr_model", "global-plates-mobile-vit-v2-model"),
        device=device,
        min_detection_conf=float(cfg.get("min_detection_conf", 0.40)),
        min_ocr_conf=float(cfg.get("min_ocr_conf", 0.50)),
    )

    tcfg = cfg.get("tracker", {})
    tracker = PlateTracker(
        iou_threshold=float(tcfg.get("iou_threshold", 0.30)),
        max_age=int(tcfg.get("max_age", 15)),
        min_hits=int(tcfg.get("min_hits", 3)),
    )

    mcfg = cfg.get("matcher", {})
    matcher = None
    if mcfg.get("enabled", False):
        matcher = PlateMatcher(
            watchlist_path=mcfg.get("watchlist", "watchlist.txt"),
            fuzzy_threshold=float(mcfg.get("fuzzy_threshold", 85)),
        )
        print(f"[info] watchlist: {len(matcher.entries)} known plate(s) loaded.")

    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        print(f"[error] could not open input: {input_src!r}")
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    writer = None
    if save_video:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    csv_writer = None
    csv_file = None
    logged: set[tuple[int, str]] = set()
    if save_csv:
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        csv_file = open(save_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "track_id", "plate", "ocr_conf", "watchlist_match"])

    print("[info] processing... (press 'q' in the window to stop)")
    frame_idx = 0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            readings = pipeline.predict(frame)
            tracker.update(readings, frame=frame, frame_idx=frame_idx)

            for track in tracker.live_tracks():
                text = track.top_text
                conf = track.top_conf
                color = GREEN
                line2 = f"{conf*100:.0f}%"

                match = matcher.match(text) if matcher else None
                if match and match.matched:
                    color = RED  # known / flagged plate
                    tag = "EXACT" if match.exact else f"~{match.score:.0f}"
                    line2 = f"MATCH {match.plate} [{tag}]"

                draw_label(frame, track.bbox, text, line2, color)

                if csv_writer is not None:
                    key = (track.id, text)
                    if key not in logged:
                        logged.add(key)
                        csv_writer.writerow(
                            [
                                frame_idx,
                                track.id,
                                text,
                                f"{conf:.3f}",
                                (match.plate if (match and match.matched) else ""),
                            ]
                        )

            # HUD
            elapsed = time.time() - t0
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                frame,
                f"frame {frame_idx} | {proc_fps:.1f} fps ({device})",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                WHITE,
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(frame)
            if show_window:
                try:
                    cv2.imshow("ALPR - press q to quit", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[info] stopped by user.")
                        break
                except cv2.error:
                    # This OpenCV build has no GUI support (headless). Keep
                    # processing and writing the output video instead.
                    print(
                        "[warn] preview window not supported by this OpenCV build; "
                        "continuing headless. The annotated video is still saved."
                    )
                    show_window = False
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    print(f"[done] processed {frame_idx} frames in {time.time() - t0:.1f}s")
    if save_video:
        print(f"[done] annotated video -> {output_path}")
    if save_csv:
        print(f"[done] detection log  -> {save_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
