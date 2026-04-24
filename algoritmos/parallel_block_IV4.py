import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from .algoritmo_base import AlgoritmoBase

BSIZE = 64

class ParallelBlockIV4(AlgoritmoBase):
    """Bloques paralelos Row×Row — cada bloque (i1,k1) en un hilo (IV.4)."""

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A); C = np.zeros((n, n))
        lock = threading.Lock()

        def tarea(i1, k1):
            tmp = np.zeros((min(BSIZE,n-i1), min(BSIZE,n-k1)))
            for j1 in range(0, n, BSIZE):
                tmp += A[i1:i1+BSIZE,j1:j1+BSIZE] @ B[j1:j1+BSIZE,k1:k1+BSIZE]
            with lock:
                C[i1:i1+BSIZE,k1:k1+BSIZE] += tmp

        with ThreadPoolExecutor() as ex:
            fs = [ex.submit(tarea,i1,k1)
                  for i1 in range(0,n,BSIZE)
                  for k1 in range(0,n,BSIZE)]
            for f in fs: f.result()
        return C    
