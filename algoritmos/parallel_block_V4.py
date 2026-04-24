import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from .algoritmo_base import AlgoritmoBase

BSIZE = 64

class ParallelBlockV4(AlgoritmoBase):
    """Bloques paralelos Col×Col — cada bloque (k1,i1) en un hilo (V.4)."""

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A); C = np.zeros((n, n))
        lock = threading.Lock()

        def tarea(k1, i1):
            tmp = np.zeros((min(BSIZE,n-k1), min(BSIZE,n-i1)))
            for j1 in range(0, n, BSIZE):
                tmp += A[k1:k1+BSIZE,j1:j1+BSIZE] @ B[j1:j1+BSIZE,i1:i1+BSIZE]
            with lock:
                C[k1:k1+BSIZE,i1:i1+BSIZE] += tmp

        with ThreadPoolExecutor() as ex:
            fs = [ex.submit(tarea,k1,i1)
                  for k1 in range(0,n,BSIZE)
                  for i1 in range(0,n,BSIZE)]
            for f in fs: f.result()
        return C
