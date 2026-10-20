import numpy as np
import pytest
from src.forward_model import default_ab2,schlumberger_curve
from src.models import ThreeLayerModel
from src.neural_core import predict_batch
from src.field_protocol import survey_geometry

def test_homogeneous():
    curve=schlumberger_curve(ThreeLayerModel(100,100,100,5,12),default_ab2())
    assert np.allclose(curve,100,rtol=1e-5)

def test_geometry():
    g=survey_geometry();assert len(g['AB2_m'])==20
    assert np.isclose(g['AB_m'][-1],400)

def test_deterministic():
    obs=schlumberger_curve(ThreeLayerModel(100,30,250,5,12),default_ab2())
    a=predict_batch(obs);b=predict_batch(obs)
    assert np.allclose(a,b,rtol=0,atol=1e-12)

def test_positive_output():
    obs=schlumberger_curve(ThreeLayerModel(100,30,250,5,12),default_ab2())
    est=predict_batch(obs)
    assert est.shape==(1,5) and np.all(est>0) and np.isfinite(est).all()

def test_invalid_input():
    with pytest.raises(ValueError):predict_batch(np.ones(19))
