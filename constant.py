import numpy as np
from dataclasses import dataclass


@dataclass
class Parameters:
    energy_tolerance: float
    electrons: int
    N: int
    dom_start: float
    dom_end: float

    def __post_init__(self):
        self.occupation = [2] * (self.electrons // 2) + ([1] if self.electrons % 2 else [])
        self.x = np.linspace(self.dom_start, self.dom_end, self.N)
        self.y = np.linspace(self.dom_start, self.dom_end, self.N)
        self.h = self.x[1] - self.x[0]  