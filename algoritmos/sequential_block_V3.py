import numpy as np
from .algoritmo_base import AlgoritmoBase

BSIZE = 64

class SequentialBlockV3(AlgoritmoBase):
    """
    Algoritmo 14: Bloques secuenciales Col×Col (sección V.3 del artículo).

    Variante donde el patrón de acceso corresponde a Col×Col: acumula
    en C[k,i] usando A[k,j] y B[j,i]. Este patrón recorre las matrices
    columna a columna. En NumPy, dado que las matrices se almacenan en
    row-major order, este acceso genera más fallos de caché que Row×Row,
    aunque el operador @ mitiga este efecto internamente.

    Complejidad temporal: O(n³)
    Complejidad espacial: O(n²)
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        C = np.zeros((n, n))
        for i1 in range(0, n, BSIZE):
            for j1 in range(0, n, BSIZE):
                for k1 in range(0, n, BSIZE):
                    C[k1:k1+BSIZE, i1:i1+BSIZE] += (
                        A[k1:k1+BSIZE, j1:j1+BSIZE]
                        @ B[j1:j1+BSIZE, i1:i1+BSIZE])
        return C