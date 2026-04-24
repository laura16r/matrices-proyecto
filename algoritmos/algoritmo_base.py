import numpy as np
 
class AlgoritmoBase:
    """Clase base para todos los algoritmos de multiplicación de matrices."""
 
    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Cada subclase debe implementar multiplicar()")
 
    def __str__(self):
        return self.__class__.__name__
 