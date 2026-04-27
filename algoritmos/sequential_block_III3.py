"""
sequential_block_III3.py
────────────────────────
Multiplicación por bloques secuencial, patrón Row×Col.
Corresponde a la sección III.3 del artículo de referencia.
"""

import numpy as np
from .algoritmo_base import AlgoritmoBase

# Tamaño del bloque — elegido para que quepa en caché L1/L2 del procesador
BSIZE = 64


class SequentialBlockIII3(AlgoritmoBase):
    """
    Algoritmo 8 — Bloques secuenciales Row×Col (sección III.3).

    En lugar de recorrer elemento por elemento, divide las matrices
    en bloques de BSIZE×BSIZE y opera bloque por bloque.

    ¿Por qué es más rápido?
        Las matrices completas (512×512 float64 = 2MB) no caben en la
        caché del procesador. Al trabajar con bloques de 64×64 (32KB),
        cada bloque sí cabe en la caché L1/L2, reduciendo drásticamente
        los accesos lentos a RAM principal.

    El operador @ de NumPy delega cada multiplicación de bloque a BLAS
    (Basic Linear Algebra Subprograms), una librería altamente optimizada
    para el hardware específico del equipo.

    Patrón de acceso: C[i1,j1] += A[i1,k1] @ B[k1,j1]  (Row×Col)

    Complejidad temporal : O(n³)
    Complejidad espacial : O(n²) — no crea matrices adicionales.
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        C = np.zeros((n, n))

        # i1: índice de inicio del bloque de filas de A y C
        for i1 in range(0, n, BSIZE):
            # j1: índice de inicio del bloque de columnas de B y C
            for j1 in range(0, n, BSIZE):
                # k1: índice de inicio del bloque compartido (columnas de A / filas de B)
                for k1 in range(0, n, BSIZE):
                    # Multiplica el bloque de A por el bloque de B y acumula en C
                    # El slicing [:] extrae el bloque correspondiente de cada matriz
                    C[i1:i1+BSIZE, j1:j1+BSIZE] += (
                        A[i1:i1+BSIZE, k1:k1+BSIZE]
                        @ B[k1:k1+BSIZE, j1:j1+BSIZE])
        return C