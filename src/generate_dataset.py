from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .models import ThreeLayerModel
from .forward_model import schlumberger_curve, default_ab2


def sample_model(rng: np.random.Generator) -> ThreeLayerModel:
    # Resistividades uniformes em log10 entre 10 e 1000 ohm.m
    rhos = 10 ** rng.uniform(np.log10(10.0), np.log10(1000.0), 3)
    h1 = rng.uniform(1.0, 20.0)
    h2 = rng.uniform(2.0, 50.0)
    return ThreeLayerModel(*rhos, h1, h2)


def generate(n: int, seed: int = 42, du: float = 2e-3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ab2 = default_ab2()
    rows = []
    for i in range(n):
        m = sample_model(rng)
        rhoa = schlumberger_curve(m, ab2, du=du)
        row = {f"rhoa_{j+1:02d}": float(v) for j, v in enumerate(rhoa)}
        row.update({"rho1": m.rho1, "rho2": m.rho2, "rho3": m.rho3, "h1": m.h1, "h2": m.h2})
        rows.append(row)
        if (i + 1) % max(1, n // 10) == 0:
            print(f"Gerados {i+1}/{n} modelos")
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--du", type=float, default=2e-3)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    df = generate(args.n, args.seed, args.du)
    path = Path(args.output or f"data/dataset_{args.n}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nDataset salvo em: {path.resolve()}")
    print(df.head())

if __name__ == "__main__":
    main()
