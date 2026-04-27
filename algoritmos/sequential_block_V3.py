"""
sequential_block_V3.py
──────────────────────
Multiplicación por bloques secuencial, patrón Col×Col.
Corresponde a la sección V.3 del artículo de referencia.
"""

import numpy as np
from .algoritmo_base import AlgoritmoBase

BSIZE = 64


class SequentialBlockV3(AlgoritmoBase):
    """
    Algoritmo 14 — Bloques secuenciales Col×Col (sección V.3).

    Variante donde el patrón de acceso corresponde al recorrido
    Col×Col del artículo:
        Acumula en C[k, i] usando A[k, j] y B[j, i]

    Diferencia con III.3 y IV.3:
        - III.3: C[i,j] += A[i,k] @ B[k,j]  (Row×Col)
        - IV.3:  C[i,k] += A[i,j] @ B[j,k]  (Row×Row)
        - V.3:   C[k,i] += A[k,j] @ B[j,i]  (Col×Col)

    El patrón Col×Col recorre las matrices columna a columna.
    En NumPy (almacenamiento row-major) este acceso genera más
    fallos de caché teóricamente, pero el operador @ lo mitiga.

    Complejidad temporal : O(n³)
    Complejidad espacial : O(n²)
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        C = np.zeros((n, n))

        for i1 in range(0, n, BSIZE):
            for j1 in range(0, n, BSIZE):
                for k1 in range(0, n, BSIZE):
                    # Patrón Col×Col: C[k,i] += A[k,j] @ B[j,i]
                    C[k1:k1+BSIZE, i1:i1+BSIZE] += (
                        A[k1:k1+BSIZE, j1:j1+BSIZE]
                        @ B[j1:j1+BSIZE, i1:i1+BSIZE])
        return C