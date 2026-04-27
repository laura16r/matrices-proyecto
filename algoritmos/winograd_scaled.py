"""
winograd_scaled.py
──────────────────
Variante del algoritmo de Winograd que escala las matrices antes
de multiplicarlas para mejorar la precisión numérica.
"""

import numpy as np
from .algoritmo_base import AlgoritmoBase
from .winograd_original import WinogradOriginal


class WinogradScaled(AlgoritmoBase):
    """
    Algoritmo 5 — Winograd con escalado previo.

    El problema con matrices de números grandes (como los de 6 dígitos
    de este proyecto) es que las operaciones intermedias pueden generar
    valores muy grandes, acumulando errores de punto flotante.

    Esta variante soluciona eso en tres pasos:
      1. Encuentra el valor absoluto máximo (lambda) de ambas matrices.
      2. Divide ambas matrices por lambda → todos los elementos quedan
         en el rango [0, 1].
      3. Aplica WinogradOriginal sobre las matrices escaladas.
      4. Deshace el escala dividiendo el resultado entre scale² (= 1/lambda²),
         ya que C = A*B implica que si A y B se escalan por s, C escala por s².

    Complejidad temporal : O(n³)
    Complejidad espacial : O(n²)
    """

    def multiplicar(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = len(A)

        # Encontrar el valor máximo absoluto entre ambas matrices
        lam = max(np.max(np.abs(A)), np.max(np.abs(B)))

        # Si ambas matrices son cero, retornar directamente
        if lam == 0:
            return np.zeros((n, n))

        # Factor de escala: lleva todos los elementos al rango [0, 1]
        scale = 1.0 / lam

        # Aplicar Winograd sobre las matrices escaladas y deshacer la escala
        return WinogradOriginal().multiplicar(A * scale, B * scale) / (scale ** 2)