"""
algoritmo_base.py
─────────────────
Define la clase base de la que heredan todos los algoritmos
de multiplicación de matrices del proyecto.

Los tamaños de matriz (n1, n2) se leen automáticamente desde
config.py, por lo que cambiarlos ahí afecta a todas las clases
sin necesidad de modificar cada archivo individualmente.
"""

import numpy as np
from config import N1, N2


class AlgoritmoBase:
    """
    Clase base abstracta para los algoritmos de multiplicación de matrices.

    Define la interfaz común que deben cumplir todas las subclases:
      - Atributos n1 y n2: tamaños de los dos casos de prueba.
      - Método multiplicar(A, B): realiza la multiplicación y retorna C.
      - Método __str__: retorna el nombre de la clase para identificarla
        al imprimir resultados y guardar el CSV.

    No se debe instanciar directamente — siempre usar una subclase.
    """

    def __init__(self):
        # n1 y n2 se heredan automáticamente en todas las subclases.
        # Para cambiar los tamaños, modificar config.py únicamente.
        self.n1 = N1
        self.n2 = N2

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Multiplica las matrices A y B y retorna la matriz resultado C.

        Parámetros:
            A (np.ndarray): matriz izquierda de dimensión n×n.
            B (np.ndarray): matriz derecha de dimensión n×n.

        Retorna:
            np.ndarray: matriz resultado C = A × B de dimensión n×n.

        Lanza:
            NotImplementedError si la subclase no implementa este método.
        """
        raise NotImplementedError("Cada subclase debe implementar multiplicar()")

    def __str__(self):
        # Retorna el nombre de la clase (ej: "NaivOnArray").
        # Se usa en ejecutar.py para identificar cada algoritmo en el CSV.
        return self.__class__.__name__