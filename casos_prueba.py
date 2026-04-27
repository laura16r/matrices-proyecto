import numpy as np
import os

def generar_caso(n, nombre_archivo):
    """
    Genera una matriz cuadrada n×n con números enteros de 6 dígitos
    (entre 100.000 y 999.999) y la guarda en disco como archivo .npy.
    Si el archivo ya existe, no lo sobreescribe (persistencia).
    """
    if not os.path.exists(nombre_archivo):
        matriz = np.random.randint(100_000, 999_999, size=(n, n))
        np.save(nombre_archivo, matriz)
        print(f"  Caso guardado: {nombre_archivo} ({n}x{n})")
    else:
        print(f"  Caso ya existe: {nombre_archivo}")

if __name__ == "__main__":
    generar_caso(1024, "caso1_A.npy")
    generar_caso(1024, "caso1_B.npy")
    generar_caso(2048, "caso2_A.npy")
    generar_caso(2048, "caso2_B.npy")