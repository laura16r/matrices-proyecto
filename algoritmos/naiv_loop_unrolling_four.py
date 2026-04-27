import numpy as np
from .algoritmo_base import AlgoritmoBase

class NaivLoopUnrollingFour(AlgoritmoBase):
    """
    Algoritmo 3: Loop unrolling de factor 4.

    Extiende la idea del unrolling de 2 procesando cuatro elementos
    por iteración. El segundo 'while' al final limpia los elementos
    restantes cuando n no es múltiplo de 4.

    Complejidad temporal: O(n³)
    Complejidad espacial: O(n²)
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                k = 0
                s = 0.0
                while k < n - 3:
                    s += (A[i,k]*B[k,j] + A[i,k+1]*B[k+1,j]
                        + A[i,k+2]*B[k+2,j] + A[i,k+3]*B[k+3,j])
                    k += 4
                while k < n:
                    s += A[i, k] * B[k, j]
                    k += 1
                C[i, j] = s
        return C