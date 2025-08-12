"""Run a minimal Flask app serving the hydride segmentation blueprint."""
from __future__ import annotations

from flask import Flask
from hydride_segmentation.api import create_blueprint


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(), url_prefix="/api/v1/hydride_segmentation")
    return app


if __name__ == "__main__":  # pragma: no cover
    create_app().run(debug=True)
