import numpy as np
import time
import csv
import os
from algoritmos import *

def medir_tiempo(func, *args):
    inicio = time.perf_counter()
    func(*args)
    return time.perf_counter() - inicio

algoritmos = [
    ("NaivOnArray",             lambda A,B: naiv_on_array(A.tolist(), B.tolist())),
    ("NaivLoopUnrollingTwo",    lambda A,B: naiv_loop_unrolling_two(A,B,len(A))),
    ("NaivLoopUnrollingFour",   lambda A,B: naiv_loop_unrolling_four(A,B,len(A))),
    ("WinogradOriginal",        lambda A,B: winograd_original(A,B,len(A))),
    ("WinogradScaled",          lambda A,B: winograd_scaled(A,B,len(A))),
    ("StrassenNaiv",            strassen_naiv),
    ("StrassenWinograd",        strassen_winograd),
    ("SequentialBlock_III3",    sequential_block_row_col),
    ("ParallelBlock_III4",      parallel_block),
    ("EnhancedParallel_III5",   enhanced_parallel_block),
    ("SequentialBlock_IV3",     sequential_block_row_row),
    ("ParallelBlock_IV4",       parallel_block),
    ("EnhancedParallel_IV5",    enhanced_parallel_block),
    ("SequentialBlock_V3",      sequential_block_col_col),
    ("ParallelBlock_V4",        parallel_block),
]

def ejecutar_todos():
    A1 = np.load("caso1_A.npy").astype(float)
    B1 = np.load("caso1_B.npy").astype(float)
    A2 = np.load("caso2_A.npy").astype(float)
    B2 = np.load("caso2_B.npy").astype(float)

    archivo = "resultados.csv"
    escribir_header = not os.path.exists(archivo)
    with open(archivo, "a", newline="") as f:
        writer = csv.writer(f)
        if escribir_header:
            writer.writerow(["Algoritmo", "Caso", "n", "Tiempo_seg"])
        for nombre, func in algoritmos:
            print(f"Ejecutando {nombre} - Caso 1 (64x64)...")
            te1 = medir_tiempo(func, A1, B1)
            writer.writerow([nombre, "Caso1", 64, round(te1, 6)])
            print(f"  TE: {te1:.4f}s")

            print(f"Ejecutando {nombre} - Caso 2 (128x128)...")
            te2 = medir_tiempo(func, A2, B2)
            writer.writerow([nombre, "Caso2", 128, round(te2, 6)])
            print(f"  TE: {te2:.4f}s")

    print("\n Resultados guardados en resultados.csv")

if __name__ == "__main__":
    ejecutar_todos()