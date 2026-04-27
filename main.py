"""
Universidad del Quindío - Ingeniería de Sistemas y Computación
Seguimiento 2: Multiplicación de Matrices Grandes

Estructura del proyecto:
    main.py             punto de entrada, corre todo
    casos_prueba.py     genera y guarda las matrices
    ejecutar.py         mide tiempos y guarda CSV
    graficar.py         genera el diagrama de barras
    algoritmos/         carpeta con las 15 clases
        __init__.py
        algoritmo_base.py
        naiv_on_array.py
        ... (un archivo por algoritmo)

Ejecutar: python main.py
"""

import os
import casos_prueba
import ejecutar
import graficar
from config import N1, N2

SEP = "─" * 55

def main():
    print(SEP)
    print("  Seguimiento 2 · Multiplicación de Matrices Grandes")
    print("  Universidad del Quindío")
    print(SEP)

    # Paso 1: generar o cargar matrices persistentes
    print("\n[1/3] Preparando casos de prueba...")
    casos_prueba.generar_caso(N1, "caso1_A.npy")
    casos_prueba.generar_caso(N1, "caso1_B.npy")
    casos_prueba.generar_caso(N2, "caso2_A.npy")
    casos_prueba.generar_caso(N2, "caso2_B.npy")

    # Paso 2: ejecutar los 15 algoritmos
    if os.path.exists("resultados.csv"):
        os.remove("resultados.csv")

    print("\n[2/3] Ejecutando 15 algoritmos...")
    ejecutar.ejecutar_todos()

    # Paso 3: generar diagrama
    print("\n[3/3] Generando diagrama de barras...")
    graficar.generar_grafica()

    print(f"\n{SEP}")
    print("  ¡Listo! Archivos generados:")
    print("    • resultados.csv")
    print("    • diagrama_tiempos.png")
    print(SEP)

if __name__ == "__main__":
    main()