import numpy as np
from .algoritmo_base import AlgoritmoBase

BSIZE = 64

class SequentialBlockIII3(AlgoritmoBase):
    """
    Algoritmo 8: Bloques secuenciales Row×Col (sección III.3 del artículo).

    Divide las matrices en bloques de BSIZE×BSIZE y realiza la multiplicación
    bloque por bloque de forma secuencial. Cada bloque cabe en la caché del
    procesador (L1/L2), lo que reduce drásticamente los accesos a RAM y
    explica su alta velocidad comparado con los métodos iterativos puros.

    El operador @ de NumPy delega cada multiplicación de bloque a BLAS,
    que está altamente optimizado para el hardware.

    Complejidad temporal: O(n³)
    Complejidad espacial: O(n²) — no crea matrices adicionales.
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        C = np.zeros((n, n))
        for i1 in range(0, n, BSIZE):
            for j1 in range(0, n, BSIZE):
                for k1 in range(0, n, BSIZE):
                    C[i1:i1+BSIZE, j1:j1+BSIZE] += (
                        A[i1:i1+BSIZE, k1:k1+BSIZE]
                        @ B[k1:k1+BSIZE, j1:j1+BSIZE])
        return C