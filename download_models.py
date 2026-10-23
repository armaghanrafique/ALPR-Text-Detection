#!/usr/bin/env python
"""
Pre-download the ALPR model weights so the first `run.py` is instant (and so you
can confirm your environment is set up correctly before processing video).

Just instantiating the pipeline triggers fast-alpr to fetch the ONNX weights
into its local cache.

Usage:
    python download_models.py
"""

import sys

import yaml


def main() -> int:
    try:
        cfg = {}
        try:
            with open("config.yaml", "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            pass

        detector = cfg.get("detector_model", "yolo-v9-s-608-license-plate-end2end")
        ocr = cfg.get("ocr_model", "global-plates-mobile-vit-v2-model")

        print(f"[info] detector model : {detector}")
        print(f"[info] ocr model      : {ocr}")
        print("[info] downloading / loading weights (this may take a minute)...")

        from alpr import PlatePipeline

        PlatePipeline(detector_model=detector, ocr_model=ocr)

        print("[ok] models are downloaded and load correctly.")
        print("[ok] you are ready to run:  python run.py --input <your_video.mp4>")
        return 0
    except ImportError as e:
        print(f"[error] missing dependency: {e}")
        print("[hint]  install requirements first:  pip install -r requirements.txt")
        return 1
    except Exception as e:  # pragma: no cover
        print(f"[error] {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
