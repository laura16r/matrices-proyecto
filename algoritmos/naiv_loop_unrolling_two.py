import numpy as np
from .algoritmo_base import AlgoritmoBase

class NaivLoopUnrollingTwo(AlgoritmoBase):
    """
    Algoritmo 2: Loop unrolling de factor 2.

    Optimiza el ciclo interno k procesando dos elementos por iteración
    en lugar de uno. Esto reduce a la mitad la cantidad de veces que
    el ciclo verifica su condición de parada. El 'if k < n' al final
    maneja el elemento sobrante cuando n es impar.

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
                while k < n - 1:
                    s += A[i, k] * B[k, j] + A[i, k+1] * B[k+1, j]
                    k += 2
                if k < n:
                    s += A[i, k] * B[k, j]
                C[i, j] = s
        return C