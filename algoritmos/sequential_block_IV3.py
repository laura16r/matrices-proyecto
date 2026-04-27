import numpy as np
from .algoritmo_base import AlgoritmoBase

BSIZE = 64

class SequentialBlockIV3(AlgoritmoBase):
    """
    Algoritmo 11: Bloques secuenciales Row×Row (sección IV.3 del artículo).

    Variante de SequentialBlockIII3 donde el orden de acceso corresponde
    al patrón Row×Row: acumula en C[i,k] usando A[i,j] y B[j,k].
    En NumPy el resultado es equivalente porque el operador @ optimiza
    el acceso internamente, pero el patrón de índices refleja el recorrido
    fila a fila sobre ambas matrices.

    Complejidad temporal: O(n³)
    Complejidad espacial: O(n²)
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        C = np.zeros((n, n))
        for i1 in range(0, n, BSIZE):
            for j1 in range(0, n, BSIZE):
                for k1 in range(0, n, BSIZE):
                    C[i1:i1+BSIZE, k1:k1+BSIZE] += (
                        A[i1:i1+BSIZE, j1:j1+BSIZE]
                        @ B[j1:j1+BSIZE, k1:k1+BSIZE])
        return C