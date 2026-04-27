import numpy as np
from config import N1, N2

class AlgoritmoBase:
    """
    Clase base para todos los algoritmos de multiplicación de matrices.
    Los tamaños n1 y n2 se leen desde config.py automáticamente.
    """

    def __init__(self):
        self.n1 = N1
        self.n2 = N2

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Cada subclase debe implementar multiplicar()")

    def __str__(self):
        return self.__class__.__name__