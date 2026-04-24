import numpy as np
from .algoritmo_base import AlgoritmoBase

class WinogradOriginal(AlgoritmoBase):
    """Winograd: reduce multiplicaciones mediante prefactorización de filas y columnas."""

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        C = np.zeros((n, n))
        row_f = np.zeros(n)
        col_f = np.zeros(n)
        for i in range(n):
            for k in range(n // 2):
                row_f[i] += A[i, 2*k] * A[i, 2*k+1]
        for j in range(n):
            for k in range(n // 2):
                col_f[j] += B[2*k, j] * B[2*k+1, j]
        for i in range(n):
            for j in range(n):
                tmp = -row_f[i] - col_f[j]
                for k in range(n // 2):
                    tmp += (A[i,2*k] + B[2*k+1,j]) * (A[i,2*k+1] + B[2*k,j])
                if n % 2 == 1:
                    tmp += A[i, n-1] * B[n-1, j]
                C[i, j] = tmp
        return C
        
