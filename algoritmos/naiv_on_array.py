import numpy as np
from .algoritmo_base import AlgoritmoBase

class NaivOnArray(AlgoritmoBase):
    """
    Algoritmo 1: Multiplicación clásica O(n³).

    Implementa los tres ciclos anidados básicos (i, j, k) trabajando
    directamente sobre listas de Python. Es el método más sencillo pero
    también el más lento para matrices grandes, ya que no aprovecha
    ninguna optimización de memoria ni paralelismo.

    Complejidad temporal: O(n³)
    Complejidad espacial: O(n²) — solo almacena la matriz resultado C.
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A_l = A.tolist()
        B_l = B.tolist()
        n = len(A_l)
        C = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i][j] += A_l[i][k] * B_l[k][j]
        return np.array(C)