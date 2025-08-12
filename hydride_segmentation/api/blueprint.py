"""Flask blueprint factory for hydride segmentation API."""
from __future__ import annotations

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from .schema import SegmentParams
from .handlers import process_request


def create_blueprint() -> Blueprint:
    bp = Blueprint("hydride_segmentation", __name__)

    @bp.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @bp.post("/segment")
    def segment_endpoint():
        try:
            params = SegmentParams(**request.form)
        except ValidationError as exc:
            detail = ", ".join([e["msg"] for e in exc.errors()])
            return jsonify({"ok": False, "error": {"code": "VALIDATION", "detail": detail}}), 400

        file = request.files.get("file")
        try:
            resp = process_request(file, params)
            return jsonify(resp)
        except ValueError as e:
            return jsonify({"ok": False, "error": {"code": "VALIDATION", "detail": str(e)}}), 400
        except Exception as e:  # pragma: no cover - unexpected
            return jsonify({"ok": False, "error": {"code": "SERVER_ERROR", "detail": str(e)}}), 500

    return bp
