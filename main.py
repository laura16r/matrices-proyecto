"""
Universidad del Quindío - Ingeniería de Sistemas y Computación
Seguimiento 2: Multiplicación de Matrices Grandes
Ejecutar: python main.py
"""

import os
import casos_prueba
import ejecutar
import graficar

SEP = "─" * 55

def main():
    print(SEP)
    print("  Seguimiento 2 · Multiplicación de Matrices Grandes")
    print("  Universidad del Quindío")
    print(SEP)

    print("\n[1/3] Preparando casos de prueba (1024×1024 y 2048×2048)...")
    casos_prueba.generar_caso(1024, "caso1_A.npy")
    casos_prueba.generar_caso(1024, "caso1_B.npy")
    casos_prueba.generar_caso(2048, "caso2_A.npy")
    casos_prueba.generar_caso(2048, "caso2_B.npy")

    if os.path.exists("resultados.csv"):
        os.remove("resultados.csv")

    print("\n[2/3] Ejecutando 15 algoritmos...")
    ejecutar.ejecutar_todos()

    print("\n[3/3] Generando diagrama de barras...")
    graficar.generar_grafica()

    print(f"\n{SEP}")
    print("  ¡Listo! Archivos generados:")
    print("    • resultados.csv")
    print("    • diagrama_tiempos.png")
    print(SEP)

if __name__ == "__main__":
    main()