"""
naiv_loop_unrolling_four.py
───────────────────────────
Extiende el loop unrolling a factor 4: procesa cuatro elementos
del ciclo k por iteración en lugar de dos.
"""

import numpy as np
from .algoritmo_base import AlgoritmoBase


class NaivLoopUnrollingFour(AlgoritmoBase):
    """
    Algoritmo 3 — Loop unrolling de factor 4.

    Procesa cuatro elementos de k por iteración:
        s += A[i,k]*B[k,j] + A[i,k+1]*B[k+1,j]
           + A[i,k+2]*B[k+2,j] + A[i,k+3]*B[k+3,j]

    El primer 'while k < n - 3' avanza de 4 en 4.
    El segundo 'while k < n' limpia los 0, 1, 2 o 3 elementos
    restantes cuando n no es múltiplo de 4.

    Complejidad temporal : O(n³)
    Complejidad espacial : O(n²)
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        C = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                k = 0
                s = 0.0  # acumulador local

                # Procesa de 4 en 4: cuatro multiplicaciones por iteración
                while k < n - 3:
                    s += (A[i,k]  * B[k,j]   + A[i,k+1] * B[k+1,j]
                        + A[i,k+2] * B[k+2,j] + A[i,k+3] * B[k+3,j])
                    k += 4

                # Limpia los elementos restantes (0 a 3 elementos)
                while k < n:
                    s += A[i, k] * B[k, j]
                    k += 1

                C[i, j] = s
        return C