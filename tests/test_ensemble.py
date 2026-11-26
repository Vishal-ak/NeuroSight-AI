import numpy as np
from src.ensemble import tune_weights, MODEL_NAMES


def test_ensemble_weight_tuning_returns_normalized_weights():
    y = np.array([0, 1, 2, 0, 1, 2])
    good = np.eye(3)[y] * 0.8 + 0.2 / 3
    weak = np.full((len(y), 3), 1 / 3)
    records = {
        MODEL_NAMES[0]: {"y_true": y, "y_prob": good},
        MODEL_NAMES[1]: {"y_true": y, "y_prob": weak},
        MODEL_NAMES[2]: {"y_true": y, "y_prob": weak},
    }
    weights, score = tune_weights(records, step=0.1)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0)
    assert score == 1.0
