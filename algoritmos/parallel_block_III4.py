import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from .algoritmo_base import AlgoritmoBase

BSIZE = 64

class ParallelBlockIII4(AlgoritmoBase):
    """
    Algoritmo 9: Bloques paralelos Row×Col (sección III.4 del artículo).

    Extiende SequentialBlockIII3 enviando cada bloque (i1, j1) como una
    tarea independiente a un pool de hilos. Un Lock protege las escrituras
    sobre la matriz C para evitar condiciones de carrera cuando dos hilos
    intentan actualizar el mismo bloque simultáneamente.

    Complejidad temporal: O(n³/p) donde p = número de hilos disponibles.
    Complejidad espacial: O(n²) + memoria de hilos del sistema.
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        C = np.zeros((n, n))
        lock = threading.Lock()

        def tarea(i1, j1):
            tmp = np.zeros((min(BSIZE, n-i1), min(BSIZE, n-j1)))
            for k1 in range(0, n, BSIZE):
                tmp += (A[i1:i1+BSIZE, k1:k1+BSIZE]
                        @ B[k1:k1+BSIZE, j1:j1+BSIZE])
            with lock:
                C[i1:i1+BSIZE, j1:j1+BSIZE] += tmp

        with ThreadPoolExecutor() as ex:
            fs = [ex.submit(tarea, i1, j1)
                  for i1 in range(0, n, BSIZE)
                  for j1 in range(0, n, BSIZE)]
            for f in fs:
                f.result()
        return C