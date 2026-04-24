import numpy as np
from .algoritmo_base import AlgoritmoBase

class StrassenWinograd(AlgoritmoBase):
    """Strassen optimizado con matrices auxiliares S y T (variante Winograd)."""

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)
        if n == 1:
            return A * B
        m = n // 2
        A11,A12 = A[:m,:m], A[:m,m:]
        A21,A22 = A[m:,:m], A[m:,m:]
        B11,B12 = B[:m,:m], B[:m,m:]
        B21,B22 = B[m:,:m], B[m:,m:]
        S1=A21+A22; T1=B12-B11
        S2=S1-A11;  T2=B22-T1
        S3=A11-A21; T3=B22-B12
        S4=A12-S2;  T4=B21-T2
        P1=self.multiplicar(A11,B11)
        P2=self.multiplicar(A12,B21)
        P3=self.multiplicar(S1,T1)
        P4=self.multiplicar(S2,T2)
        P5=self.multiplicar(S3,T3)
        P6=self.multiplicar(S4,B22)
        P7=self.multiplicar(A22,T4)
        U1=P1+P2; U2=P1+P4; U3=U2+P5
        C = np.zeros((n, n))
        C[:m,:m] = U1
        C[:m,m:] = U3+P3+P6
        C[m:,:m] = U2-P3+P7
        C[m:,m:] = U3+P7
        return C
        
