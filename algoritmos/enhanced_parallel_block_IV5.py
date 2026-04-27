"""
enhanced_parallel_block_IV5.py
───────────────────────────────
Versión mejorada del paralelismo por bloques Row×Row con 2 hilos.
Corresponde a la sección IV.5 del artículo de referencia.
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .algoritmo_base import AlgoritmoBase

BSIZE = 64


class EnhancedParallelBlockIV5(AlgoritmoBase):
    """
    Algoritmo 13 — Bloques paralelos mejorados Row×Row (sección IV.5).

    Aplica la misma estrategia de 2 hilos de EnhancedParallelBlockIII5
    pero con el patrón de acceso Row×Row de la sección IV del artículo.

    La matriz se divide horizontalmente en dos mitades iguales:
        - Hilo 1: procesa filas 0 a mid con patrón Row×Row
        - Hilo 2: procesa filas mid a n con patrón Row×Row

    Al usar solo 2 hilos se evita el overhead de crear y sincronizar
    muchos hilos pequeños, logrando mejor rendimiento en la práctica.

    Complejidad temporal : O(n³/2)
    Complejidad espacial : O(n²)
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        C = np.zeros((n, n))
        mid = n // 2

        def mitad(inicio, fin):
            for i1 in range(inicio, fin, BSIZE):
                for j1 in range(0, n, BSIZE):
                    for k1 in range(0, n, BSIZE):
                        # Patrón Row×Row: C[i,k] += A[i,j] @ B[j,k]
                        C[i1:i1+BSIZE, k1:k1+BSIZE] += (
                            A[i1:i1+BSIZE, j1:j1+BSIZE]
                            @ B[j1:j1+BSIZE, k1:k1+BSIZE])

        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(mitad, 0, mid)
            f2 = ex.submit(mitad, mid, n)
            f1.result()
            f2.result()
        return C