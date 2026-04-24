import numpy as np
from .algoritmo_base import AlgoritmoBase

BSIZE = 64

class SequentialBlockV3(AlgoritmoBase):
    """Bloques secuenciales Col×Col (artículo sección V.3)."""

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A); C = np.zeros((n, n))
        for i1 in range(0, n, BSIZE):
            for j1 in range(0, n, BSIZE):
                for k1 in range(0, n, BSIZE):
                    C[k1:k1+BSIZE,i1:i1+BSIZE] += (
                        A[k1:k1+BSIZE,j1:j1+BSIZE]
                        @ B[j1:j1+BSIZE,i1:i1+BSIZE])
        return C
