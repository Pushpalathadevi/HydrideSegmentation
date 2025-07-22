import base64
import io
import os
import tempfile
from typing import Tuple

from flask import Flask, jsonify, request, send_file
from PIL import Image
import numpy as np

from .inference import run_model as run_ml_model
from .segmentation_mask_creation import run_model as run_conv_model
from .core.analysis import orientation_analysis

DEFAULT_PARAMS = {
    "clahe": {"clip_limit": 2.0, "tile_grid_size": [8, 8]},
    "adaptive": {"block_size": 13, "C": 20},
    "morph": {"kernel_size": [5, 5], "iterations": 0},
    "area_threshold": 150,
    "crop": False,
    "crop_percent": 0,
}

app = Flask(__name__)


def _save_upload(file) -> str:
    """Save uploaded file to a temporary location and return path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    file.save(tmp.name)
    return tmp.name


def _segment(
    image_path: str, model: str, params: dict
) -> Tuple[np.ndarray, np.ndarray]:
    if model == "ml":
        return run_ml_model(image_path, params)
    return run_conv_model(image_path, params)


def _overlay(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        rgb = np.stack([image] * 3, axis=-1)
    else:
        rgb = image.copy()
    rgb[mask > 0] = [255, 0, 0]
    return Image.fromarray(rgb)


@app.route("/infer", methods=["POST"])
def infer():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    model = request.form.get("model", "ml").lower()
    analysis = request.form.get("analysis", "false").lower() == "true"

    params = DEFAULT_PARAMS.copy()
    if model != "ml":
        for key in params:
            if key in request.form:
                try:
                    params[key] = eval(request.form[key])
                except Exception:
                    pass

    tmp_path = _save_upload(file)
    try:
        image, mask = _segment(tmp_path, model, params)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    mask_img = Image.fromarray(mask)

    if not analysis:
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    input_img = Image.fromarray(image if image.ndim == 2 else image)
    overlay_img = _overlay(image, mask)
    orient_img, size_img, angle_img = orientation_analysis(mask)

    buf_mask = io.BytesIO()
    mask_img.save(buf_mask, format="PNG")
    buf_mask.seek(0)
    buf_input = io.BytesIO()
    input_img.save(buf_input, format="PNG")
    buf_input.seek(0)
    buf_overlay = io.BytesIO()
    overlay_img.save(buf_overlay, format="PNG")
    buf_overlay.seek(0)
    buf_orient = io.BytesIO()
    orient_img.save(buf_orient, format="PNG")
    buf_orient.seek(0)
    buf_size = io.BytesIO()
    size_img.save(buf_size, format="PNG")
    buf_size.seek(0)
    buf_angle = io.BytesIO()
    angle_img.save(buf_angle, format="PNG")
    buf_angle.seek(0)

    fraction = float(np.count_nonzero(mask) / mask.size)
    return jsonify(
        {
            "original": base64.b64encode(buf_input.getvalue()).decode(),
            "mask": base64.b64encode(buf_mask.getvalue()).decode(),
            "overlay": base64.b64encode(buf_overlay.getvalue()).decode(),
            "orientation": base64.b64encode(buf_orient.getvalue()).decode(),
            "size_distribution": base64.b64encode(buf_size.getvalue()).decode(),
            "angle_distribution": base64.b64encode(buf_angle.getvalue()).decode(),
            "area_fraction": fraction,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def main() -> None:
    app.run(host="0.0.0.0", port=5004)


if __name__ == "__main__":
    main()
