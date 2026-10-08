"""Modelagem direta 1D para SEV Schlumberger, modelo de 3 camadas.

Implementa a formulação descrita no projeto do TCC:
  T3 = rho3
  Ti = (T_{i+1} + rho_i*tanh(lambda*h_i)) /
       (1 + (T_{i+1}/rho_i)*tanh(lambda*h_i))

e
  rho_a(s) = rho1 + s^2 * integral([T(lambda)-rho1] J1(lambda*s) lambda dlambda)

A integral é feita em u=ln(lambda), por trapézios.
"""
from __future__ import annotations
import numpy as np
from scipy.special import j1
from .models import ThreeLayerModel


def resistivity_transform(lam: np.ndarray, m: ThreeLayerModel) -> np.ndarray:
    t = np.full_like(lam, m.rho3, dtype=float)
    for rho, h in ((m.rho2, m.h2), (m.rho1, m.h1)):
        th = np.tanh(lam * h)
        t = (t + rho * th) / (1.0 + (t / rho) * th)
    return t


def schlumberger_curve(
    m: ThreeLayerModel,
    ab2: np.ndarray,
    u_min: float = np.log(1e-4),
    u_max: float = np.log(12.0),
    du: float = 2e-3,
) -> np.ndarray:
    """Calcula rho_a para cada AB/2.

    du=2e-3 é um padrão mais rápido para exploração. Para reprodução final
    do TCC, use du=5e-4 (mais lento), conforme a metodologia escrita.
    """
    u = np.arange(u_min, u_max + du, du)
    lam = np.exp(u)
    t = resistivity_transform(lam, m)
    delta = t - m.rho1
    out = []
    # dlambda = lambda du; integrando original tem *lambda dlambda,
    # portanto em u aparece lambda^2 du.
    for s in np.asarray(ab2, dtype=float):
        integrand_u = delta * j1(lam * s) * (lam ** 2)
        integral = np.trapezoid(integrand_u, u)
        out.append(m.rho1 + (s ** 2) * integral)
    return np.asarray(out)


def default_ab2(n: int = 20) -> np.ndarray:
    return np.logspace(np.log10(1.0), np.log10(200.0), n)
