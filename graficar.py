import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generar_grafica():
    df = pd.read_csv("resultados.csv")

    # Limpiar filas que sean duplicados del encabezado
    df = df[df["Caso"] != "Caso"]
    df["Tiempo_seg"] = df["Tiempo_seg"].astype(float)

    caso1 = df[df["Caso"] == "Caso1"]
    caso2 = df[df["Caso"] == "Caso2"]

    etiquetas = [n.split(" - ")[0] for n in caso1["Algoritmo"]]
    x = np.arange(len(etiquetas))
    w = 0.38

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#F5F7FF")
    ax.set_facecolor("#F5F7FF")

    b1 = ax.bar(x - w/2, caso1["Tiempo_seg"].values, w,
                label="Caso 1 (64×64)", color="#2A52B0", edgecolor="#0D1A3A", linewidth=0.6)
    b2 = ax.bar(x + w/2, caso2["Tiempo_seg"].values, w,
                label="Caso 2 (128×128)", color="#6B7DA0", edgecolor="#0D1A3A", linewidth=0.6)

    ax.bar_label(b1, fmt="%.3f", fontsize=7, padding=2, color="#0D1A3A")
    ax.bar_label(b2, fmt="%.3f", fontsize=7, padding=2, color="#0D1A3A")

    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Tiempo de ejecución (segundos)", fontsize=11)
    ax.set_title("Multiplicación de Matrices Grandes — Comparación de Algoritmos\n"
                 "Universidad del Quindío · Ingeniería de Sistemas",
                 fontsize=13, fontweight="bold", color="#0D1A3A")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("diagrama_tiempos.png", dpi=150, bbox_inches="tight")
    print(" Diagrama guardado en 'diagrama_tiempos.png'")
    plt.show()