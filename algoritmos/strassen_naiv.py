import numpy as np
from .algoritmo_base import AlgoritmoBase

class StrassenNaiv(AlgoritmoBase):
    """
    Algoritmo 6: Strassen clásico (divide y vencerás).

    Divide cada matriz en 4 submatrices de n/2 × n/2 y calcula
    el resultado usando solo 7 productos recursivos (P1 a P7) en lugar
    de los 8 del método clásico. Las submatrices del resultado se
    reconstruyen combinando los P con sumas y restas.

    Caso base: cuando n==1, multiplica directamente los dos escalares.

    Complejidad temporal: O(n^2.807)
    Complejidad espacial: O(n² log n) — la recursión crea submatrices
                          en cada nivel, lo que aumenta el uso de RAM.
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        if n == 1:
            return A * B
        m = n // 2
        A11, A12 = A[:m, :m], A[:m, m:]
        A21, A22 = A[m:, :m], A[m:, m:]
        B11, B12 = B[:m, :m], B[:m, m:]
        B21, B22 = B[m:, :m], B[m:, m:]

        P1 = self.multiplicar(A11+A22, B11+B22)
        P2 = self.multiplicar(A21+A22, B11)
        P3 = self.multiplicar(A11,     B12-B22)
        P4 = self.multiplicar(A22,     B21-B11)
        P5 = self.multiplicar(A11+A12, B22)
        P6 = self.multiplicar(A21-A11, B11+B12)
        P7 = self.multiplicar(A12-A22, B21+B22)

        C = np.zeros((n, n))
        C[:m, :m] = P1 + P4 - P5 + P7
        C[:m, m:] = P3 + P5
        C[m:, :m] = P2 + P4
        C[m:, m:] = P1 - P2 + P3 + P6
        return C