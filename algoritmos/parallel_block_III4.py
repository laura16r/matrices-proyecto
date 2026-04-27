"""
parallel_block_III4.py
──────────────────────
Versión paralela del algoritmo de bloques Row×Col.
Cada bloque (i1, j1) se procesa en un hilo independiente.
Corresponde a la sección III.4 del artículo de referencia.
"""

import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from .algoritmo_base import AlgoritmoBase

BSIZE = 64


class ParallelBlockIII4(AlgoritmoBase):
    """
    Algoritmo 9 — Bloques paralelos Row×Col (sección III.4).

    Extiende SequentialBlockIII3 distribuyendo los bloques entre
    múltiples hilos del sistema usando ThreadPoolExecutor.

    Funcionamiento:
        - Se crea una función 'tarea(i1, j1)' que calcula un bloque completo
          de C acumulando sobre k1.
        - Se envían todas las tareas al pool con executor.submit().
        - El Lock (threading.Lock) protege las escrituras sobre C para
          evitar condiciones de carrera: si dos hilos intentan escribir
          en el mismo bloque simultáneamente, el Lock garantiza que solo
          uno lo haga a la vez.
        - f.result() espera a que cada tarea termine antes de retornar.

    Complejidad temporal : O(n³/p) donde p = hilos disponibles del sistema.
    Complejidad espacial : O(n²) + memoria de los hilos del sistema.
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        C = np.zeros((n, n))

        # Lock para proteger escrituras concurrentes sobre C
        lock = threading.Lock()

        def tarea(i1, j1):
            # Cada hilo calcula su bloque de forma independiente
            # en una variable temporal 'tmp' para minimizar el tiempo
            # que mantiene el lock ocupado
            tmp = np.zeros((min(BSIZE, n-i1), min(BSIZE, n-j1)))
            for k1 in range(0, n, BSIZE):
                tmp += (A[i1:i1+BSIZE, k1:k1+BSIZE]
                        @ B[k1:k1+BSIZE, j1:j1+BSIZE])
            # Solo al final escribe en C, minimizando el tiempo bloqueado
            with lock:
                C[i1:i1+BSIZE, j1:j1+BSIZE] += tmp

        # Enviar todos los bloques al pool de hilos
        with ThreadPoolExecutor() as ex:
            fs = [ex.submit(tarea, i1, j1)
                  for i1 in range(0, n, BSIZE)
                  for j1 in range(0, n, BSIZE)]
            # Esperar a que todos los hilos terminen
            for f in fs:
                f.result()
        return C