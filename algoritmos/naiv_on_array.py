import numpy as np
from .algoritmo_base import AlgoritmoBase

class NaivOnArray(AlgoritmoBase):
    """Multiplicación clásica O(n³) con listas puras de Python."""

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A_l = A.tolist(); B_l = B.tolist()
        n = len(A_l)
        C = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i][j] += A_l[i][k] * B_l[k][j]
        return np.array(C)