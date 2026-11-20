
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ARCHIVO = BASE / "data" / "processed" / "games_clusters.csv"
SALIDA = BASE / "outputs" / "graficas"
DATOS = BASE / "data" / "processed"

SALIDA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ARCHIVO)

# 1. Estadisticas descriptivas
variables = [
    "price",
    "Owners_Estimados",
    "Total_Resenas",
    "Porcentaje_Positivo",
    "average_playtime"
]

print("\n===== ESTADISTICAS POR GRUPO =====")

estadisticas = df.groupby("Grupo")[variables].agg(
    ["mean", "median", "min", "max"]
)

print(estadisticas.round(2).to_string())

estadisticas.to_csv(
    DATOS / "estadisticas_grupos.csv"
)

# 2. Cantidad de videojuegos
print("\n===== DISTRIBUCION =====")
print(df["Grupo"].value_counts().sort_index())

# 3. Generos mas frecuentes
print("\n===== GENEROS POR GRUPO =====")

generos = (
    df.assign(Genero=df["genres"].str.split(";"))
    .explode("Genero")
    .groupby(["Grupo", "Genero"])
    .size()
    .reset_index(name="Cantidad")
)

top_generos = (
    generos.sort_values(
        ["Grupo", "Cantidad"],
        ascending=[True, False]
    )
    .groupby("Grupo")
    .head(5)
)

print(top_generos.to_string(index=False))

top_generos.to_csv(
    DATOS / "generos_grupos.csv",
    index=False
)

# 4. Comparacion de propietarios
propietarios = df.groupby("Grupo")[
    "Owners_Estimados"
].median()

plt.figure(figsize=(9, 5))

propietarios.plot(
    kind="bar",
    color=["steelblue", "darkorange"]
)

plt.title("Mediana de propietarios estimados por grupo")
plt.xlabel("Grupo")
plt.ylabel("Propietarios estimados")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    SALIDA / "07_propietarios_grupos.png",
    dpi=150
)

plt.show()

# 5. Comparacion de valoraciones
valoraciones = df.groupby("Grupo")[
    "Porcentaje_Positivo"
].mean()

plt.figure(figsize=(9, 5))

valoraciones.plot(
    kind="bar",
    color=["steelblue", "darkorange"]
)

plt.title("Promedio de valoraciones positivas por grupo")
plt.xlabel("Grupo")
plt.ylabel("Porcentaje positivo")
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    SALIDA / "08_valoraciones_grupos.png",
    dpi=150
)

plt.show()

print("\nAnalisis de grupos finalizado.")
