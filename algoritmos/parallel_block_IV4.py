"""
parallel_block_IV4.py
─────────────────────
Versión paralela del algoritmo de bloques Row×Row.
Corresponde a la sección IV.4 del artículo de referencia.
"""

import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from .algoritmo_base import AlgoritmoBase

BSIZE = 64


class ParallelBlockIV4(AlgoritmoBase):
    """
    Algoritmo 12 — Bloques paralelos Row×Row (sección IV.4).

    Versión paralela de SequentialBlockIV3. Cada par de bloques
    (i1, k1) se asigna a un hilo independiente del pool.

    El Lock protege las escrituras sobre C[i1, k1] para evitar
    condiciones de carrera entre hilos que actualicen la misma
    posición simultáneamente.

    La variable temporal 'tmp' permite que cada hilo calcule su
    bloque completo antes de adquirir el lock, minimizando el
    tiempo de espera entre hilos.

    Complejidad temporal : O(n³/p)
    Complejidad espacial : O(n²) + memoria de hilos del sistema.
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        C = np.zeros((n, n))
        lock = threading.Lock()

        def tarea(i1, k1):
            # Calcular el bloque completo en tmp antes de escribir en C
            tmp = np.zeros((min(BSIZE, n-i1), min(BSIZE, n-k1)))
            for j1 in range(0, n, BSIZE):
                tmp += (A[i1:i1+BSIZE, j1:j1+BSIZE]
                        @ B[j1:j1+BSIZE, k1:k1+BSIZE])
            # Escribir en C de forma segura con el lock
            with lock:
                C[i1:i1+BSIZE, k1:k1+BSIZE] += tmp

        with ThreadPoolExecutor() as ex:
            fs = [ex.submit(tarea, i1, k1)
                  for i1 in range(0, n, BSIZE)
                  for k1 in range(0, n, BSIZE)]
            for f in fs:
                f.result()
        return C