import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from .algoritmo_base import AlgoritmoBase

BSIZE = 64

class ParallelBlockIII4(AlgoritmoBase):
    """Bloques paralelos Row×Col — cada bloque (i1,j1) en un hilo (III.4)."""

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A); C = np.zeros((n, n))
        lock = threading.Lock()

        def tarea(i1, j1):
            tmp = np.zeros((min(BSIZE,n-i1), min(BSIZE,n-j1)))
            for k1 in range(0, n, BSIZE):
                tmp += A[i1:i1+BSIZE,k1:k1+BSIZE] @ B[k1:k1+BSIZE,j1:j1+BSIZE]
            with lock:
                C[i1:i1+BSIZE,j1:j1+BSIZE] += tmp

        with ThreadPoolExecutor() as ex:
            fs = [ex.submit(tarea,i1,j1)
                  for i1 in range(0,n,BSIZE)
                  for j1 in range(0,n,BSIZE)]
            for f in fs: f.result()
        return C
