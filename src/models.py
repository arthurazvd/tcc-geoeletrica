from dataclasses import dataclass

@dataclass(frozen=True)
class ThreeLayerModel:
    rho1: float
    rho2: float
    rho3: float
    h1: float
    h2: float

    def as_array(self):
        return [self.rho1, self.rho2, self.rho3, self.h1, self.h2]
