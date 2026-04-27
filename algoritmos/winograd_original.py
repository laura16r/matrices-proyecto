import numpy as np
from .algoritmo_base import AlgoritmoBase

class WinogradOriginal(AlgoritmoBase):
    """
    Algoritmo 4: Método de Winograd original.

    Reduce el número de multiplicaciones mediante una prefactorización
    en tres fases:
      Fase 1 — calcula row_factor[i]: producto de pares consecutivos por fila de A.
      Fase 2 — calcula col_factor[j]: producto de pares consecutivos por columna de B.
      Fase 3 — usa row_factor y col_factor para calcular cada C[i][j] con
                menos multiplicaciones que el método clásico.

    Si n es impar, se agrega el último elemento sobrante manualmente.

    Complejidad temporal: O(n³)
    Complejidad espacial: O(n²) + O(n) para row_factor y col_factor.
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        C = np.zeros((n, n))
        row_f = np.zeros(n)
        col_f = np.zeros(n)

        # Fase 1: prefactorización de filas de A
        for i in range(n):
            for k in range(n // 2):
                row_f[i] += A[i, 2*k] * A[i, 2*k+1]

        # Fase 2: prefactorización de columnas de B
        for j in range(n):
            for k in range(n // 2):
                col_f[j] += B[2*k, j] * B[2*k+1, j]

        # Fase 3: cálculo del resultado usando los factores precalculados
        for i in range(n):
            for j in range(n):
                tmp = -row_f[i] - col_f[j]
                for k in range(n // 2):
                    tmp += (A[i,2*k] + B[2*k+1,j]) * (A[i,2*k+1] + B[2*k,j])
                if n % 2 == 1:
                    tmp += A[i, n-1] * B[n-1, j]
                C[i, j] = tmp
        return C