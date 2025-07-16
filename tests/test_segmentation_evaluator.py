import numpy as np
from hydride_segmentation.segmentation_evaluator import SegmentationEvaluator


def test_tamper_mask_deterministic():
    np.random.seed(0)
    evaluator = SegmentationEvaluator({'simulate': True, 'plot': False})
    mask = np.ones((10, 10), dtype=np.uint8)
    tampered = evaluator.tamper_mask(mask, 0.1)
    diff = np.sum(mask != tampered)
    # 10% of 100 pixels should be flipped
    assert diff == 10


def test_compute_metrics_simulated():
    np.random.seed(0)
    evaluator = SegmentationEvaluator({'simulate': True, 'plot': False})
    evaluator.generate_simulated_data()
    metrics = evaluator.compute_metrics()
    assert set(['IoU', 'Precision', 'Recall', 'F1 Score', 'Dice Coefficient', 'MSE', 'Accuracy']).issubset(metrics.keys())
    for v in metrics.values():
        assert 0.0 <= v <= 1.0 or v >= 0  # mse can be >1

