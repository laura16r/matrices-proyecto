import numpy as np
import time
import csv
import os

from algoritmos import (
    NaivOnArray, NaivLoopUnrollingTwo, NaivLoopUnrollingFour,
    WinogradOriginal, WinogradScaled,
    StrassenNaiv, StrassenWinograd,
    SequentialBlockIII3, ParallelBlockIII4, EnhancedParallelBlockIII5,
    SequentialBlockIV3,  ParallelBlockIV4,  EnhancedParallelBlockIV5,
    SequentialBlockV3,   ParallelBlockV4,
)

ALGORITMOS = [
    NaivOnArray(),
    NaivLoopUnrollingTwo(),
    NaivLoopUnrollingFour(),
    WinogradOriginal(),
    WinogradScaled(),
    StrassenNaiv(),
    StrassenWinograd(),
    SequentialBlockIII3(),
    ParallelBlockIII4(),
    EnhancedParallelBlockIII5(),
    SequentialBlockIV3(),
    ParallelBlockIV4(),
    EnhancedParallelBlockIV5(),
    SequentialBlockV3(),
    ParallelBlockV4(),
]

def medir_tiempo(algoritmo, A, B):
    inicio = time.perf_counter()
    algoritmo.multiplicar(A, B)
    return time.perf_counter() - inicio

def ejecutar_todos():
    A1 = np.load("caso1_A.npy").astype(float)
    B1 = np.load("caso1_B.npy").astype(float)
    A2 = np.load("caso2_A.npy").astype(float)
    B2 = np.load("caso2_B.npy").astype(float)
    n1, n2 = len(A1), len(A2)

    archivo = "resultados.csv"
    escribir_header = not os.path.exists(archivo)
    with open(archivo, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if escribir_header:
            writer.writerow(["Algoritmo", "Caso", "n", "Tiempo_seg"])
        for algo in ALGORITMOS:
            nombre = str(algo)
            print(f"  {nombre} - Caso 1 ({n1}x{n1})...")
            te1 = medir_tiempo(algo, A1, B1)
            writer.writerow([nombre, "Caso1", n1, round(te1, 6)])
            print(f"    TE: {te1:.4f}s")

            print(f"  {nombre} - Caso 2 ({n2}x{n2})...")
            te2 = medir_tiempo(algo, A2, B2)
            writer.writerow([nombre, "Caso2", n2, round(te2, 6)])
            print(f"    TE: {te2:.4f}s")

    print("\n  ✅ Resultados guardados en resultados.csv")

if __name__ == "__main__":
    ejecutar_todos()