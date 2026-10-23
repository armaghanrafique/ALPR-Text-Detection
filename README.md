# License Plate Recognition (ALPR)

Reads vehicle number plates from a video (or webcam), draws the detected number
on screen, logs every plate, and can flag plates that match a watchlist of known
plates using precise + fuzzy text matching.

Runs on **CPU** out of the box and moves to **GPU** later with a one-line change.

## Stack (most accurate pretrained models)

| Stage              | Model                                              | Notes                              |
|--------------------|----------------------------------------------------|------------------------------------|
| Plate detection    | YOLOv9 (`yolo-v9-s-608-license-plate-end2end`)     | via `fast-alpr` / open-image-models|
| Plate OCR          | MobileViT (`global-plates-mobile-vit-v2-model`)    | via `fast-plate-ocr`               |
| Stabilisation      | IOU tracker + confidence-weighted voting           | `alpr/tracker.py`                  |
| Precise matching   | Normalised exact + `rapidfuzz` fuzzy match         | `alpr/matcher.py`                  |

Weights download automatically on first run.

## Setup (CPU / local)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt

# Optional: pre-download weights and verify the install
python download_models.py
```

## Web app (localhost)

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser. Type a video path (or `0` for the
webcam) and press **Start**.

- **Left pane:** live annotated video.
- **Right pane:** one card per vehicle that passes — the cropped plate image, the
  recognised number, and its confidence. The number shown is the **single
  most-confident reading** for that car, finalised when the car leaves the frame.
- Watchlist matches are highlighted in **red**.

## Run (command line)

```bash
# Process a video file
python run.py --input path/to/your_video.mp4

# Use the webcam
python run.py --input 0

# Headless (no preview window) - just write the annotated video + CSV
python run.py --input path/to/your_video.mp4 --no-show
```

Outputs land in `output/`:
- `annotated.mp4` — video with plate numbers drawn on each vehicle
- `detections.csv` — `frame, track_id, plate, ocr_conf, watchlist_match`

Press **q** in the preview window to stop early.

## Precise text matching

Put known plates (one per line) in `watchlist.txt`. Any read plate that matches
one of them is drawn in **red** with the canonical plate text. Matching is
case/space-insensitive and folds common OCR confusions (O↔0, I↔1, S↔5, B↔8, …).
Tune strictness with `matcher.fuzzy_threshold` in `config.yaml` (100 = exact only).

## Tuning (`config.yaml`)

- **More accuracy:** keep `detector_model: yolo-v9-s-608-license-plate-end2end`.
- **More speed:** switch to `yolo-v9-t-384-license-plate-end2end` (or `-256`).
- **Fewer false reads:** raise `min_detection_conf` / `min_ocr_conf`.
- **Steadier display:** raise `tracker.min_hits`; increase `max_age` for occlusions.

## Moving to GPU later

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

Then set `device: cuda` in `config.yaml` (or run `python run.py --device cuda`).
No other changes needed.

## Project layout

```
config.yaml          # all settings
requirements.txt     # dependencies (CPU)
download_models.py   # pre-fetch + verify weights
run.py               # main entry point (video -> overlay + CSV)
watchlist.txt        # known plates for precise matching
alpr/
  pipeline.py        # detection + OCR (fast-alpr wrapper)
  tracker.py         # IOU tracking + reading aggregation
  matcher.py         # watchlist exact/fuzzy matching
```
