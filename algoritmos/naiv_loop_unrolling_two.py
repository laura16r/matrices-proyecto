import numpy as np
from .algoritmo_base import AlgoritmoBase

class NaivLoopUnrollingTwo(AlgoritmoBase):
    """Loop unrolling procesando 2 elementos de k por iteración."""

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A); C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                k = 0; s = 0.0
                while k < n - 1:
                    s += A[i,k]*B[k,j] + A[i,k+1]*B[k+1,j]
                    k += 2
                if k < n:
                    s += A[i,k] * B[k,j]
                C[i,j] = s
        return C
        
