#!/usr/bin/env python
"""
Flask web app for the license plate reader.

Run:
    python app.py
Then open:  http://127.0.0.1:5000

Left pane  = live annotated video.
Right pane = finalised plates (best crop image + recognised text + confidence),
             one card per vehicle that passes.
"""

from __future__ import annotations

import os
import time

import yaml
from flask import Flask, Response, jsonify, render_template, request

from engine import Engine


def load_config(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


app = Flask(__name__)
CFG = load_config()
ENGINE = Engine(CFG)


@app.route("/")
def index():
    default_source = str(CFG.get("input", "0"))
    return render_template("index.html", default_source=default_source)


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(silent=True) or {}
    source = (data.get("source") or str(CFG.get("input", "0"))).strip()
    msg = ENGINE.start(source)
    return jsonify({"ok": True, "message": msg, "source": source})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    return jsonify({"ok": True, "message": ENGINE.stop()})


@app.route("/api/state")
def api_state():
    return jsonify(ENGINE.get_state())


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json(silent=True) or {}
    return jsonify({"ok": True, "settings": ENGINE.update_settings(data)})


def _mjpeg():
    boundary = b"--frame"
    blank_sent = False
    while True:
        jpeg = ENGINE.get_jpeg()
        if jpeg is None:
            if blank_sent:
                time.sleep(0.1)
                continue
            blank_sent = True
        else:
            blank_sent = False
            yield (
                boundary
                + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
        time.sleep(0.04)  # ~25 fps cap on the stream


@app.route("/video_feed")
def video_feed():
    return Response(
        _mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    # use_reloader=False so the heavy models/engine load only once.
    app.run(host="127.0.0.1", port=5000, threaded=True, use_reloader=False)
