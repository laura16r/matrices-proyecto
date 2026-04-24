import numpy as np
from .algoritmo_base import AlgoritmoBase

BSIZE = 64

class SequentialBlockIV3(AlgoritmoBase):
    """Bloques secuenciales Row×Row (artículo sección IV.3)."""

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A); C = np.zeros((n, n))
        for i1 in range(0, n, BSIZE):
            for j1 in range(0, n, BSIZE):
                for k1 in range(0, n, BSIZE):
                    C[i1:i1+BSIZE,k1:k1+BSIZE] += (
                        A[i1:i1+BSIZE,j1:j1+BSIZE]
                        @ B[j1:j1+BSIZE,k1:k1+BSIZE])
        return C
