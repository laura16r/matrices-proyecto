"""
sequential_block_IV3.py
───────────────────────
Multiplicación por bloques secuencial, patrón Row×Row.
Corresponde a la sección IV.3 del artículo de referencia.
"""

import numpy as np
from .algoritmo_base import AlgoritmoBase

BSIZE = 64


class SequentialBlockIV3(AlgoritmoBase):
    """
    Algoritmo 11 — Bloques secuenciales Row×Row (sección IV.3).

    Variante de SequentialBlockIII3 donde el patrón de acceso
    corresponde al recorrido Row×Row del artículo:
        Acumula en C[i, k] usando A[i, j] y B[j, k]

    Diferencia con III.3 (Row×Col):
        - III.3 accede a C[i,j], A[i,k], B[k,j]
        - IV.3  accede a C[i,k], A[i,j], B[j,k]

    En la práctica con NumPy el rendimiento es similar porque el
    operador @ optimiza el acceso a memoria internamente. La diferencia
    es conceptual y refleja el orden de recorrido del artículo original.

    Complejidad temporal : O(n³)
    Complejidad espacial : O(n²)
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        C = np.zeros((n, n))

        for i1 in range(0, n, BSIZE):
            for j1 in range(0, n, BSIZE):
                for k1 in range(0, n, BSIZE):
                    # Patrón Row×Row: C[i,k] += A[i,j] @ B[j,k]
                    C[i1:i1+BSIZE, k1:k1+BSIZE] += (
                        A[i1:i1+BSIZE, j1:j1+BSIZE]
                        @ B[j1:j1+BSIZE, k1:k1+BSIZE])
        return C