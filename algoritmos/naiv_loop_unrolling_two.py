"""
naiv_loop_unrolling_two.py
──────────────────────────
Optimización del algoritmo clásico mediante loop unrolling de factor 2:
en vez de procesar un elemento de k por iteración, procesa dos a la vez.
"""

import numpy as np
from .algoritmo_base import AlgoritmoBase


class NaivLoopUnrollingTwo(AlgoritmoBase):
    """
    Algoritmo 2 — Loop unrolling de factor 2.

    El loop unrolling reduce la cantidad de veces que el ciclo evalúa
    su condición de parada, lo que en lenguajes compilados mejora el
    rendimiento. En Python el beneficio es limitado por el overhead
    del intérprete, pero ilustra la técnica correctamente.

    El ciclo 'while k < n - 1' avanza de 2 en 2, procesando:
        s += A[i,k]*B[k,j] + A[i,k+1]*B[k+1,j]

    El 'if k < n' al final maneja el elemento sobrante cuando n es impar.

    Complejidad temporal : O(n³)
    Complejidad espacial : O(n²)
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        C = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                k = 0
                s = 0.0  # acumulador local para evitar escrituras repetidas en C

                # Procesa de 2 en 2: dos multiplicaciones por iteración
                while k < n - 1:
                    s += A[i, k] * B[k, j] + A[i, k+1] * B[k+1, j]
                    k += 2

                # Si n es impar queda un elemento sin procesar, se agrega aquí
                if k < n:
                    s += A[i, k] * B[k, j]

                C[i, j] = s
        return C