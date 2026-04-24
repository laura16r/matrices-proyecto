import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .algoritmo_base import AlgoritmoBase

BSIZE = 64

class EnhancedParallelBlockIV5(AlgoritmoBase):
    """Bloques paralelos mejorados Row×Row: 2 hilos, mitad de filas cada uno (IV.5)."""

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A); C = np.zeros((n, n)); mid = n // 2

        def mitad(inicio, fin):
            for i1 in range(inicio, fin, BSIZE):
                for j1 in range(0, n, BSIZE):
                    for k1 in range(0, n, BSIZE):
                        C[i1:i1+BSIZE,k1:k1+BSIZE] += (
                            A[i1:i1+BSIZE,j1:j1+BSIZE]
                            @ B[j1:j1+BSIZE,k1:k1+BSIZE])

        with ThreadPoolExecutor(max_workers=2) as ex:
            f1=ex.submit(mitad,0,mid); f2=ex.submit(mitad,mid,n)
            f1.result(); f2.result()
        return C
