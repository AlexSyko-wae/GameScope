
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Rutas
BASE = Path(__file__).resolve().parent.parent
ENTRADA = BASE / "data" / "raw" / "steam.csv"
SALIDA = BASE / "data" / "processed"

SALIDA.mkdir(parents=True, exist_ok=True)

# 1. Leer datos
df = pd.read_csv(ENTRADA)

print("===== DATOS ORIGINALES =====")
print("Registros:", len(df))
print("Columnas:", len(df.columns))

# 2. Revisar valores nulos
print("\n===== VALORES NULOS =====")
print(df.isnull().sum())

# Rellenar datos descriptivos faltantes
df["developer"] = df["developer"].fillna("Desconocido")
df["publisher"] = df["publisher"].fillna("Desconocido")

# 3. Eliminar duplicados
df = df.drop_duplicates(subset=["appid"]).copy()

# 4. Corregir tipos de datos
df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)

columnas_numericas = [
    "price",
    "positive_ratings",
    "negative_ratings",
    "average_playtime"
]

for columna in columnas_numericas:
    df[columna] = pd.to_numeric(
        df[columna],
        errors="coerce"
    )

# 5. Convertir rango de propietarios
rangos = df["owners"].str.split("-", expand=True)

df["Owners_Min"] = pd.to_numeric(
    rangos[0], errors="coerce"
)

df["Owners_Max"] = pd.to_numeric(
    rangos[1], errors="coerce"
)

df["Owners_Estimados"] = (
    df["Owners_Min"] + df["Owners_Max"]
) / 2

# 6. Crear nuevas variables
df["Total_Resenas"] = (
    df["positive_ratings"] + df["negative_ratings"]
)

df["Porcentaje_Positivo"] = np.where(
    df["Total_Resenas"] > 0,
    df["positive_ratings"] / df["Total_Resenas"] * 100,
    0
)

df["Anio_Lanzamiento"] = df["release_date"].dt.year

# 7. Seleccionar variables para K-Means
variables = [
    "price",
    "Owners_Estimados",
    "Total_Resenas",
    "Porcentaje_Positivo",
    "average_playtime"
]

# Eliminar registros invalidos para el modelo
df = df.dropna(subset=variables).copy()

for columna in variables:
    df = df[df[columna] >= 0]

# 8. Identificar valores atipicos mediante IQR
print("\n===== VALORES ATIPICOS =====")

for columna in variables:
    q1 = df[columna].quantile(0.25)
    q3 = df[columna].quantile(0.75)
    iqr = q3 - q1

    inferior = q1 - 1.5 * iqr
    superior = q3 + 1.5 * iqr

    atipicos = (
        (df[columna] < inferior) |
        (df[columna] > superior)
    ).sum()

    print(columna, ":", atipicos)

# 9. Transformacion logaritmica
# Reduce la asimetria de variables con valores grandes
variables_log = [
    "price",
    "Owners_Estimados",
    "Total_Resenas",
    "average_playtime"
]

datos_modelo = df[variables].copy()

for columna in variables_log:
    datos_modelo[columna] = np.log1p(
        datos_modelo[columna]
    )

# 10. Normalizar con StandardScaler
escalador = StandardScaler()
datos_normalizados = escalador.fit_transform(datos_modelo)

df_normalizado = pd.DataFrame(
    datos_normalizados,
    columns=variables,
    index=df.index
)

# 11. Mostrar resultados
print("\n===== DATOS LIMPIOS =====")
print("Registros:", len(df))
print("Nulos restantes:", df.isnull().sum().sum())

print("\n===== VARIABLES NORMALIZADAS =====")
print(df_normalizado.head(10).round(3))

# 12. Guardar archivos
df.to_csv(
    SALIDA / "steam_limpio.csv",
    index=False
)

df_normalizado.to_csv(
    SALIDA / "steam_normalizado.csv",
    index=False
)

print("\nArchivos guardados correctamente.")
print("Limpieza y preparacion finalizadas.")
