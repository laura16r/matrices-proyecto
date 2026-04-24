import numpy as np
from .algoritmo_base import AlgoritmoBase
from .winograd_original import WinogradOriginal

class WinogradScaled(AlgoritmoBase):
    """Winograd con escalado previo para mejorar precisión numérica."""

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        lam = max(np.max(np.abs(A)), np.max(np.abs(B)))
        if lam == 0:
            return np.zeros((n, n))
        s = 1.0 / lam
        return WinogradOriginal().multiplicar(A * s, B * s) / (s ** 2)
