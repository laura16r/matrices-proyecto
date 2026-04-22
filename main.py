"""
Universidad del Quindío - Ingeniería de Sistemas y Computación
Seguimiento 2: Multiplicación de Matrices Grandes

Estructura esperada:
    main.py
    casos_prueba.py
    algoritmos.py
    ejecutar.py
    graficar.py

Ejecutar: python main.py
"""

import casos_prueba
import ejecutar
import graficar

def main():
    separador = "─" * 50
    print(separador)
    print("  Seguimiento 2 · Multiplicación de Matrices")
    print("  Universidad del Quindío")
    print(separador)

    print("\n[1/3] Generando / cargando casos de prueba...")
    casos_prueba.generar_caso(64,  "caso1_A.npy")
    casos_prueba.generar_caso(64,  "caso1_B.npy")
    casos_prueba.generar_caso(128, "caso2_A.npy")
    casos_prueba.generar_caso(128, "caso2_B.npy")

    print("\n[2/3] Ejecutando algoritmos y guardando tiempos...")
    ejecutar.ejecutar_todos()

    print("\n[3/3] Generando diagrama de barras...")
    graficar.generar_grafica()

    print(f"\n{separador}")
    print("  ¡Listo! Archivos generados:")
    print("    • resultados.csv")
    print("    • diagrama_tiempos.png")
    print(separador)

if __name__ == "__main__":
    main()