"""
strassen_naiv.py
────────────────
Implementa el algoritmo de Strassen (1969): divide y vencerás con
7 productos recursivos en lugar de 8, logrando O(n^2.807).
"""

import numpy as np
from .algoritmo_base import AlgoritmoBase


class StrassenNaiv(AlgoritmoBase):
    """
    Algoritmo 6 — Strassen clásico (divide y vencerás).

    Idea principal:
        La multiplicación clásica de matrices 2×2 requiere 8 productos.
        Strassen demostró en 1969 que se puede hacer con solo 7 productos
        (P1 a P7) combinados con sumas y restas.

        Al aplicar esto recursivamente, la complejidad baja de O(n³) a
        O(n^log₂7) ≈ O(n^2.807).

    Funcionamiento:
        1. Caso base: si n==1, multiplica directamente los dos escalares.
        2. Dividir: parte cada matriz en 4 submatrices de n/2 × n/2.
        3. Calcular 7 productos recursivos (P1 a P7).
        4. Combinar: reconstruye las 4 submatrices del resultado con sumas/restas.

    Nota: requiere que n sea potencia de 2 para que las divisiones
    recursivas lleguen siempre a matrices enteras.

    Complejidad temporal : O(n^2.807)
    Complejidad espacial : O(n² log n) — la recursión crea submatrices
                           en cada nivel, aumentando el uso de RAM.
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)

        # Caso base: matriz 1×1, multiplicación escalar directa
        if n == 1:
            return A * B

        # Punto medio para dividir la matriz en 4 cuadrantes
        m = n // 2

        # Dividir A en 4 submatrices
        A11, A12 = A[:m, :m], A[:m, m:]  # fila superior
        A21, A22 = A[m:, :m], A[m:, m:]  # fila inferior

        # Dividir B en 4 submatrices
        B11, B12 = B[:m, :m], B[:m, m:]
        B21, B22 = B[m:, :m], B[m:, m:]

        # 7 productos recursivos de Strassen (en lugar de 8 del método clásico)
        P1 = self.multiplicar(A11 + A22, B11 + B22)  # (A11+A22)(B11+B22)
        P2 = self.multiplicar(A21 + A22, B11)         # (A21+A22)B11
        P3 = self.multiplicar(A11,       B12 - B22)   # A11(B12-B22)
        P4 = self.multiplicar(A22,       B21 - B11)   # A22(B21-B11)
        P5 = self.multiplicar(A11 + A12, B22)         # (A11+A12)B22
        P6 = self.multiplicar(A21 - A11, B11 + B12)   # (A21-A11)(B11+B12)
        P7 = self.multiplicar(A12 - A22, B21 + B22)   # (A12-A22)(B21+B22)

        # Reconstruir la matriz resultado combinando los 7 productos
        C = np.zeros((n, n))
        C[:m, :m] = P1 + P4 - P5 + P7  # cuadrante superior izquierdo
        C[:m, m:] = P3 + P5             # cuadrante superior derecho
        C[m:, :m] = P2 + P4             # cuadrante inferior izquierdo
        C[m:, m:] = P1 - P2 + P3 + P6  # cuadrante inferior derecho
        return C