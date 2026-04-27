"""
strassen_winograd.py
────────────────────
Variante de Strassen que usa matrices auxiliares S y T para
reutilizar sumas entre los 7 productos, reduciendo operaciones.
"""

import numpy as np
from .algoritmo_base import AlgoritmoBase


class StrassenWinograd(AlgoritmoBase):
    """
    Algoritmo 7 — Strassen con la optimización de Winograd.

    Mejora a StrassenNaiv precalculando 8 matrices auxiliares (S1-S4, T1-T4)
    antes de los 7 productos recursivos. Estas matrices reutilizan sumas
    entre productos, reduciendo las operaciones de suma/resta de 18 a 15
    comparado con el Strassen clásico.

    Adicionalmente, calcula 3 combinaciones intermedias (U1, U2, U3) que
    se reutilizan al armar los cuadrantes del resultado final, evitando
    recalcular las mismas sumas varias veces.

    Funcionamiento:
        S1..S4 son combinaciones de submatrices de A.
        T1..T4 son combinaciones de submatrices de B.
        P1..P7 son los 7 productos recursivos (usando S y T).
        U1..U3 son combinaciones intermedias del resultado.

    Complejidad temporal : O(n^2.807)
    Complejidad espacial : O(n² log n) — mayor que StrassenNaiv por
                           las matrices auxiliares S, T y U adicionales.
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)

        # Caso base: matriz 1×1
        if n == 1:
            return A * B

        m = n // 2

        # Dividir en submatrices
        A11, A12 = A[:m, :m], A[:m, m:]
        A21, A22 = A[m:, :m], A[m:, m:]
        B11, B12 = B[:m, :m], B[:m, m:]
        B21, B22 = B[m:, :m], B[m:, m:]

        # Matrices auxiliares S (combinaciones de A) y T (combinaciones de B)
        # Se precalculan para reutilizarlas en los productos P
        S1 = A21 + A22;  T1 = B12 - B11
        S2 = S1  - A11;  T2 = B22 - T1
        S3 = A11 - A21;  T3 = B22 - B12
        S4 = A12 - S2;   T4 = B21 - T2

        # 7 productos recursivos usando las matrices auxiliares
        P1 = self.multiplicar(A11, B11)
        P2 = self.multiplicar(A12, B21)
        P3 = self.multiplicar(S1,  T1)
        P4 = self.multiplicar(S2,  T2)
        P5 = self.multiplicar(S3,  T3)
        P6 = self.multiplicar(S4,  B22)
        P7 = self.multiplicar(A22, T4)

        # Combinaciones intermedias reutilizadas en los cuadrantes de C
        U1 = P1 + P2
        U2 = P1 + P4
        U3 = U2 + P5

        # Reconstruir la matriz resultado
        C = np.zeros((n, n))
        C[:m, :m] = U1
        C[:m, m:] = U3 + P3 + P6
        C[m:, :m] = U2 - P3 + P7
        C[m:, m:] = U3 + P7
        return C