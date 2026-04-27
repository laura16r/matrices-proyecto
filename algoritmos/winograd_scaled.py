import numpy as np
from .algoritmo_base import AlgoritmoBase
from .winograd_original import WinogradOriginal

class WinogradScaled(AlgoritmoBase):
    """
    Algoritmo 5: Winograd con escalado previo.

    Antes de aplicar WinogradOriginal, escala ambas matrices dividiéndolas
    por su valor absoluto máximo (lambda). Esto lleva todos los elementos
    al rango [0, 1], mejorando la precisión numérica cuando las matrices
    contienen números muy grandes (como los de 6 dígitos de este proyecto).
    Al final, deshace el escalado dividiendo entre scale².

    Complejidad temporal: O(n³)
    Complejidad espacial: O(n²)
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        lam = max(np.max(np.abs(A)), np.max(np.abs(B)))
        if lam == 0:
            return np.zeros((n, n))
        scale = 1.0 / lam
        return WinogradOriginal().multiplicar(
            A * scale, B * scale) / (scale ** 2)