
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Rutas del proyecto
BASE = Path(__file__).resolve().parent.parent
ARCHIVO = BASE / "data" / "raw" / "steam.csv"
SALIDA = BASE / "outputs" / "graficas"

SALIDA.mkdir(parents=True, exist_ok=True)

# 1. Cargar el dataset
df = pd.read_csv(ARCHIVO)

print("\n===== PRIMEROS 10 VIDEOJUEGOS =====")
print(df.head(10).to_string(index=False))

# 2. Dimensiones y columnas
print("\n===== DIMENSIONES =====")
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])

print("\n===== COLUMNAS =====")
print(df.columns.tolist())

# 3. Tipos de datos
print("\n===== TIPOS DE DATOS =====")
print(df.dtypes)

# 4. Estadisticas descriptivas
print("\n===== ESTADISTICAS =====")
print(df.describe().round(2))

# 5. Valores nulos
print("\n===== VALORES NULOS =====")
print(df.isnull().sum())

# 6. Registros duplicados
print("\n===== DUPLICADOS =====")
print("Filas duplicadas:", df.duplicated().sum())
print("AppID duplicados:", df["appid"].duplicated().sum())

# 7. Grafica de precios
plt.figure(figsize=(10, 5))

plt.hist(
    df["price"],
    bins=50,
    color="steelblue",
    edgecolor="black"
)

plt.title("Distribucion de precios de videojuegos")
plt.xlabel("Precio registrado")
plt.ylabel("Cantidad de videojuegos")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(SALIDA / "01_precios.png", dpi=150)
plt.show()

# 8. Grafica de generos
generos = (
    df["genres"]
    .dropna()
    .str.split(";")
    .explode()
    .value_counts()
    .head(10)
)

plt.figure(figsize=(11, 6))
generos.sort_values().plot.barh(color="seagreen")

plt.title("10 generos mas frecuentes")
plt.xlabel("Cantidad de videojuegos")
plt.ylabel("Genero")
plt.tight_layout()
plt.savefig(SALIDA / "02_generos.png", dpi=150)
plt.show()

# 9. Grafica de valoraciones
df["Total_Resenas"] = (
    df["positive_ratings"] + df["negative_ratings"]
)

plt.figure(figsize=(10, 6))

plt.scatter(
    df["positive_ratings"],
    df["negative_ratings"],
    alpha=0.35,
    s=12,
    color="darkorange"
)

plt.xscale("symlog", linthresh=1)
plt.yscale("symlog", linthresh=1)

plt.title("Valoraciones positivas y negativas")
plt.xlabel("Valoraciones positivas")
plt.ylabel("Valoraciones negativas")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SALIDA / "03_valoraciones.png", dpi=150)
plt.show()

print("\nExploracion terminada.")
print("Graficas guardadas en:", SALIDA)
