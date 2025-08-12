from io import BytesIO

import numpy as np
from PIL import Image
from flask import Flask
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from hydride_segmentation.api import create_blueprint


def _create_app():
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(), url_prefix="/api/v1/hydride_segmentation")
    return app


def _test_image_bytes() -> bytes:
    img = np.zeros((50, 50), dtype=np.uint8)
    rr, cc = np.ogrid[:50, :50]
    mask = (rr - 25) ** 2 + (cc - 15) ** 2 <= 25
    img[mask] = 255
    img[10:40, 30:40] = 255
    with BytesIO() as buf:
        Image.fromarray(img).save(buf, format="PNG")
        return buf.getvalue()


def test_health_endpoint():
    app = _create_app()
    client = app.test_client()
    res = client.get('/api/v1/hydride_segmentation/health')
    assert res.status_code == 200
    assert res.get_json() == {"status": "ok"}


def test_segment_endpoint_conventional_and_ml():
    app = _create_app()
    client = app.test_client()
    img_bytes = _test_image_bytes()
    data = {'file': (BytesIO(img_bytes), 'test.png')}
    res = client.post('/api/v1/hydride_segmentation/segment', data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    out = res.get_json()
    assert out['ok'] is True
    assert out['model'] == 'conventional'
    assert 'mask_png_b64' in out['images']

    data_ml = {'file': (BytesIO(img_bytes), 'test.png'), 'model': 'ml'}
    res_ml = client.post('/api/v1/hydride_segmentation/segment', data=data_ml, content_type='multipart/form-data')
    assert res_ml.status_code == 200
    out_ml = res_ml.get_json()
    assert out_ml['model'] == 'ml'
    assert out_ml['images'] == {}
    assert out_ml['metrics'] == {}
