"""
naiv_on_array.py
────────────────
Implementa el algoritmo de multiplicación de matrices más básico:
tres ciclos anidados que recorren todas las combinaciones de i, j, k.
"""

import numpy as np
from .algoritmo_base import AlgoritmoBase


class NaivOnArray(AlgoritmoBase):
    """
    Algoritmo 1 — Multiplicación clásica O(n³).

    Es la implementación directa de la definición matemática:
        C[i][j] = suma de A[i][k] * B[k][j] para todo k

    Trabaja sobre listas puras de Python (no NumPy) para reflejar
    fielmente el comportamiento del algoritmo sin optimizaciones
    externas. Es el método más lento para matrices grandes.

    Complejidad temporal : O(n³)
    Complejidad espacial : O(n²) — solo almacena C en memoria.
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # Convertir a listas de Python para operar sin NumPy internamente
        A_l = A.tolist()
        B_l = B.tolist()
        n = len(A_l)

        # Inicializar la matriz resultado con ceros
        C = [[0.0] * n for _ in range(n)]

        # Ciclo i: recorre cada fila de A (y de C)
        for i in range(n):
            # Ciclo j: recorre cada columna de B (y de C)
            for j in range(n):
                # Ciclo k: calcula el producto punto entre
                # la fila i de A y la columna j de B
                for k in range(n):
                    C[i][j] += A_l[i][k] * B_l[k][j]

        # Convertir el resultado de vuelta a numpy array antes de retornar
        return np.array(C)