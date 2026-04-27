"""
enhanced_parallel_block_III5.py
────────────────────────────────
Versión mejorada del paralelismo por bloques: en lugar de crear
un hilo por bloque, usa exactamente 2 hilos dividiendo la matriz
en mitad superior e inferior. Corresponde a la sección III.5.
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .algoritmo_base import AlgoritmoBase

BSIZE = 64


class EnhancedParallelBlockIII5(AlgoritmoBase):
    """
    Algoritmo 10 — Bloques paralelos mejorados Row×Col (sección III.5).

    Mejora a ParallelBlockIII4 reduciendo el overhead de hilos.
    En vez de crear un hilo por cada bloque (que puede ser cientos),
    crea exactamente 2 hilos:
        - Hilo 1: procesa las filas 0 a mid (mitad superior de C)
        - Hilo 2: procesa las filas mid a n (mitad inferior de C)

    Ventaja sobre III.4:
        Crear y sincronizar muchos hilos pequeños genera overhead.
        Con solo 2 hilos que procesan porciones grandes, el overhead
        es mínimo y el trabajo está bien distribuido.

    f1.result() y f2.result() bloquean hasta que ambos hilos terminen
    para garantizar que C esté completa antes de retornarla.

    Complejidad temporal : O(n³/2)
    Complejidad espacial : O(n²)
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        C = np.zeros((n, n))
        mid = n // 2  # punto de división entre los dos hilos

        def mitad(inicio, fin):
            # Cada hilo procesa las filas desde 'inicio' hasta 'fin'
            for i1 in range(inicio, fin, BSIZE):
                for j1 in range(0, n, BSIZE):
                    for k1 in range(0, n, BSIZE):
                        C[i1:i1+BSIZE, j1:j1+BSIZE] += (
                            A[i1:i1+BSIZE, k1:k1+BSIZE]
                            @ B[k1:k1+BSIZE, j1:j1+BSIZE])

        # Lanzar exactamente 2 hilos con max_workers=2
        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(mitad, 0, mid)    # hilo 1: filas superiores
            f2 = ex.submit(mitad, mid, n)    # hilo 2: filas inferiores
            f1.result()  # esperar hilo 1
            f2.result()  # esperar hilo 2
        return C