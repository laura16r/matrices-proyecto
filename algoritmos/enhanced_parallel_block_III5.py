import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .algoritmo_base import AlgoritmoBase

BSIZE = 64

class EnhancedParallelBlockIII5(AlgoritmoBase):
    """
    Algoritmo 10: Bloques paralelos mejorados Row×Col (sección III.5 del artículo).

    En lugar de crear un hilo por bloque como en III.4, usa exactamente
    2 hilos: uno procesa la mitad superior de las filas y otro la mitad
    inferior. Esto evita el overhead de crear y sincronizar decenas de
    hilos, resultando más eficiente en la práctica.

    f1.result() y f2.result() bloquean hasta que ambos hilos terminen
    antes de retornar C.

    Complejidad temporal: O(n³/2)
    Complejidad espacial: O(n²)
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        C = np.zeros((n, n))
        mid = n // 2

        def mitad(inicio, fin):
            for i1 in range(inicio, fin, BSIZE):
                for j1 in range(0, n, BSIZE):
                    for k1 in range(0, n, BSIZE):
                        C[i1:i1+BSIZE, j1:j1+BSIZE] += (
                            A[i1:i1+BSIZE, k1:k1+BSIZE]
                            @ B[k1:k1+BSIZE, j1:j1+BSIZE])

        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(mitad, 0, mid)
            f2 = ex.submit(mitad, mid, n)
            f1.result()
            f2.result()
        return C