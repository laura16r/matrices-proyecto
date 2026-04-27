"""
winograd_original.py
────────────────────
Implementa el algoritmo de Winograd, que reduce el número de
multiplicaciones mediante una prefactorización de filas y columnas.
"""

import numpy as np
from .algoritmo_base import AlgoritmoBase


class WinogradOriginal(AlgoritmoBase):
    """
    Algoritmo 4 — Winograd Original.

    Funciona en tres fases para reducir el número de multiplicaciones:

    FASE 1 — Precalcular row_factor[i]:
        Para cada fila i de A, multiplica pares de elementos consecutivos:
        row_factor[i] = A[i,0]*A[i,1] + A[i,2]*A[i,3] + ...
        Se calcula una sola vez y se reutiliza para todas las columnas de B.

    FASE 2 — Precalcular col_factor[j]:
        Para cada columna j de B, multiplica pares de filas consecutivas:
        col_factor[j] = B[0,j]*B[1,j] + B[2,j]*B[3,j] + ...

    FASE 3 — Calcular C usando los factores precalculados:
        C[i,j] = -row_factor[i] - col_factor[j]
                 + suma de (A[i,2k] + B[2k+1,j]) * (A[i,2k+1] + B[2k,j])
        Si n es impar, agrega el último elemento sobrante.

    El truco algebraico permite calcular dos productos con una sola
    multiplicación real, reduciendo el total de n³ a aproximadamente n³/2.

    Complejidad temporal : O(n³)
    Complejidad espacial : O(n²) + O(n) para row_factor y col_factor.
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        C = np.zeros((n, n))

        # Vectores de prefactorización — se calculan una sola vez
        row_f = np.zeros(n)  # un valor por fila de A
        col_f = np.zeros(n)  # un valor por columna de B

        # FASE 1: producto de pares consecutivos por fila de A
        for i in range(n):
            for k in range(n // 2):
                row_f[i] += A[i, 2*k] * A[i, 2*k+1]

        # FASE 2: producto de pares consecutivos por columna de B
        for j in range(n):
            for k in range(n // 2):
                col_f[j] += B[2*k, j] * B[2*k+1, j]

        # FASE 3: calcular cada elemento de C usando los factores
        for i in range(n):
            for j in range(n):
                # Inicio con los factores negativos (parte de la fórmula de Winograd)
                tmp = -row_f[i] - col_f[j]

                # Suma los productos combinados de pares de A y B
                for k in range(n // 2):
                    tmp += (A[i,2*k] + B[2*k+1,j]) * (A[i,2*k+1] + B[2*k,j])

                # Si n es impar, agrega el último elemento que quedó sin par
                if n % 2 == 1:
                    tmp += A[i, n-1] * B[n-1, j]

                C[i, j] = tmp
        return C