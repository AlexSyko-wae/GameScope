import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. RUTAS DEL PROYECTO
# ==========================================

BASE = Path(__file__).resolve().parent.parent

RUTA_LIMPIO = BASE / "data" / "processed" / "steam_limpio.csv"
RUTA_NORMALIZADO = BASE / "data" / "processed" / "steam_normalizado.csv"

SALIDA = BASE / "outputs" / "graficas"
SALIDA.mkdir(parents=True, exist_ok=True)

# Cargar archivos
df = pd.read_csv(RUTA_LIMPIO)
x = pd.read_csv(RUTA_NORMALIZADO)

if len(df) != len(x):
    raise ValueError("Los archivos tienen diferente cantidad de registros")

print("===== DATOS CARGADOS =====")
print("Videojuegos:", len(df))
print("Variables:", x.columns.tolist())

# ==========================================
# 2. EVALUAR DIFERENTES VALORES DE K
# ==========================================

valores_k = range(2, 7)

inercias = []
siluetas = []

for k in valores_k:

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    etiquetas = modelo.fit_predict(x)

    inercia = modelo.inertia_

    # Muestra para evitar calculos demasiado costosos
    silueta = silhouette_score(
        x,
        etiquetas,
        sample_size=min(3000, len(x)),
        random_state=42
    )

    inercias.append(inercia)
    siluetas.append(silueta)

    print(f"K={k} | Inercia={inercia:.2f} | Silueta={silueta:.4f}")

# ==========================================
# 3. METODO DEL CODO
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    list(valores_k),
    inercias,
    marker="o"
)

plt.title("Metodo del codo - GameScope")
plt.xlabel("Numero de clusters (K)")
plt.ylabel("Inercia")
plt.xticks(list(valores_k))
plt.grid(True)

plt.savefig(
    SALIDA / "04_metodo_codo.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# 4. COEFICIENTE DE SILUETA
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    list(valores_k),
    siluetas,
    marker="o",
    color="green"
)

plt.title("Coeficiente de silueta - GameScope")
plt.xlabel("Numero de clusters (K)")
plt.ylabel("Puntuacion de silueta")
plt.xticks(list(valores_k))
plt.grid(True)

plt.savefig(
    SALIDA / "05_silueta.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# 5. SELECCIONAR K
# ==========================================

# Seleccion inicial: mayor puntuacion de silueta.
# Se revisara tambien el codo y la interpretacion.
mejor_indice = np.argmax(siluetas)
mejor_k = list(valores_k)[mejor_indice]

print("\n===== SELECCION DEL MODELO =====")
print("K con mejor silueta:", mejor_k)

# ==========================================
# 6. ENTRENAR MODELO FINAL
# ==========================================

modelo_final = KMeans(
    n_clusters=mejor_k,
    random_state=42,
    n_init=10
)

df["Grupo"] = modelo_final.fit_predict(x)

print("\n===== VIDEOJUEGOS AGRUPADOS =====")
print(
    df[["name", "price", "Grupo"]]
    .head(20)
    .to_string(index=False)
)

print("\n===== CANTIDAD POR GRUPO =====")
print(df["Grupo"].value_counts().sort_index())

print("\n===== CENTROIDES NORMALIZADOS =====")
centroides = pd.DataFrame(
    modelo_final.cluster_centers_,
    columns=x.columns
)
print(centroides.round(3))

# ==========================================
# 7. CENTROIDES EN ESCALA ORIGINAL
# ==========================================

variables_log = [
    "price",
    "Owners_Estimados",
    "Total_Resenas",
    "average_playtime"
]

# Reconstruir la transformacion utilizada
datos_transformados = df[x.columns].copy()

for columna in variables_log:
    datos_transformados[columna] = np.log1p(
        datos_transformados[columna]
    )

escalador = StandardScaler()
escalador.fit(datos_transformados)

centroides_originales = pd.DataFrame(
    escalador.inverse_transform(
        modelo_final.cluster_centers_
    ),
    columns=x.columns
)

for columna in variables_log:
    centroides_originales[columna] = np.expm1(
        centroides_originales[columna]
    )

print("\n===== CENTROIDES INTERPRETABLES =====")
print(centroides_originales.round(2))

# ==========================================
# 8. ITERACIONES Y DISTANCIAS
# ==========================================

print("\nNumero de iteraciones:")
print(modelo_final.n_iter_)

print("\nDistancias de los primeros 10 juegos:")
print(
    pd.DataFrame(
        modelo_final.transform(x),
        columns=[
            f"Centroide_{i}"
            for i in range(mejor_k)
        ]
    ).head(10).round(3)
)

# ==========================================
# 9. GRAFICA DE AGRUPAMIENTOS
# ==========================================

plt.figure(figsize=(11, 6))

plt.scatter(
    x["price"],
    x["Total_Resenas"],
    c=df["Grupo"],
    cmap="viridis",
    alpha=0.35,
    s=12
)

plt.scatter(
    modelo_final.cluster_centers_[
        :, x.columns.get_loc("price")
    ],
    modelo_final.cluster_centers_[
        :, x.columns.get_loc("Total_Resenas")
    ],
    color="red",
    marker="X",
    s=250,
    label="Centroides"
)

plt.title(f"Segmentacion de videojuegos - K={mejor_k}")
plt.xlabel("Precio normalizado")
plt.ylabel("Resenas normalizadas")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(
    SALIDA / "06_clusters.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# 10. COMPARAR DOS CONFIGURACIONES
# ==========================================

print("\n===== COMPARACION K=2 Y K=3 =====")

for k in [2, 3]:
    comparacion = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    grupos = comparacion.fit_predict(x)

    print(f"\nModelo K={k}")
    print("Inercia:", round(comparacion.inertia_, 2))
    print("Iteraciones:", comparacion.n_iter_)
    print("Cantidad por grupo:")
    print(pd.Series(grupos).value_counts().sort_index())

# ==========================================
# 11. GUARDAR RESULTADOS
# ==========================================

df.to_csv(
    BASE / "data" / "processed" / "games_clusters.csv",
    index=False
)

centroides_originales.to_csv(
    BASE / "data" / "processed" / "centroides.csv",
    index_label="Grupo"
)

pd.DataFrame({
    "K": list(valores_k),
    "Inercia": inercias,
    "Silueta": siluetas
}).to_csv(
    BASE / "data" / "processed" / "evaluacion_k.csv",
    index=False
)

print("\nGAME SCOPE: MODELO FINALIZADO")
print("Archivos y graficas guardados.")
