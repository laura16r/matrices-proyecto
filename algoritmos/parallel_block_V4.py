import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from .algoritmo_base import AlgoritmoBase

BSIZE = 64

class ParallelBlockV4(AlgoritmoBase):
    """
    Algoritmo 15: Bloques paralelos Col×Col (sección V.4 del artículo).

    Versión paralela de SequentialBlockV3. Cada par de bloques (k1, i1)
    se asigna a un hilo independiente del pool. El Lock protege las
    escrituras sobre C[k1, i1] para evitar condiciones de carrera.

    Complejidad temporal: O(n³/p)
    Complejidad espacial: O(n²) + memoria de hilos del sistema.
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        C = np.zeros((n, n))
        lock = threading.Lock()

        def tarea(k1, i1):
            tmp = np.zeros((min(BSIZE, n-k1), min(BSIZE, n-i1)))
            for j1 in range(0, n, BSIZE):
                tmp += (A[k1:k1+BSIZE, j1:j1+BSIZE]
                        @ B[j1:j1+BSIZE, i1:i1+BSIZE])
            with lock:
                C[k1:k1+BSIZE, i1:i1+BSIZE] += tmp

        with ThreadPoolExecutor() as ex:
            fs = [ex.submit(tarea, k1, i1)
                  for k1 in range(0, n, BSIZE)
                  for i1 in range(0, n, BSIZE)]
            for f in fs:
                f.result()
        return C