from pathlib import Path
import matplotlib.pyplot as plt
from .models import ThreeLayerModel
from .forward_model import schlumberger_curve, default_ab2


def main():
    # Mesmo modelo ilustrativo descrito no TCC
    model = ThreeLayerModel(rho1=100, rho2=30, rho3=250, h1=5, h2=12)
    ab2 = default_ab2()
    rhoa = schlumberger_curve(model, ab2)

    print("Modelo: [rho1, rho2, rho3, h1, h2]")
    print(model.as_array())
    print("\nAB/2 (m) -> rho aparente (ohm.m)")
    for x, y in zip(ab2, rhoa):
        print(f"{x:8.3f} -> {y:10.3f}")

    Path("results/figures").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.loglog(ab2, rhoa, "o-")
    plt.xlabel("AB/2 (m)")
    plt.ylabel("Resistividade aparente (ohm.m)")
    plt.title("SEV sintética - modelo de 3 camadas")
    plt.grid(True, which="both", alpha=.25)
    plt.tight_layout()
    out = Path("results/figures/demo_sev.png")
    plt.savefig(out, dpi=160)
    print(f"\nGráfico salvo em {out}")

if __name__ == "__main__":
    main()
