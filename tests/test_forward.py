import numpy as np
from src.models import ThreeLayerModel
from src.forward_model import schlumberger_curve, default_ab2

def test_homogeneous_medium():
    m = ThreeLayerModel(100, 100, 100, 5, 12)
    curve = schlumberger_curve(m, default_ab2(), du=2e-3)
    assert np.allclose(curve, 100.0, rtol=1e-8, atol=1e-8)
