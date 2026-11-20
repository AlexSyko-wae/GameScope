
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="GameScope",
    page_icon="🎮",
    layout="wide"
)

BASE = Path(__file__).resolve().parent.parent
ARCHIVO = BASE / "data" / "processed" / "games_clusters.csv"
EVALUACION = BASE / "data" / "processed" / "evaluacion_k.csv"

NOMBRES = {
    0: "Baja actividad registrada",
    1: "Alta participación"
}

@st.cache_data
def cargar_datos():
    datos = pd.read_csv(ARCHIVO)
    datos["release_date"] = pd.to_datetime(
        datos["release_date"], errors="coerce"
    )
    datos["Anio_Lanzamiento"] = datos["release_date"].dt.year
    datos["Segmento"] = datos["Grupo"].map(NOMBRES)
    datos["Segmento"] = datos["Segmento"].fillna(
        "Grupo " + datos["Grupo"].astype(str)
    )
    return datos

df = cargar_datos()

st.title("🎮 GameScope")
st.caption(
    "Sistema de análisis y segmentación de videojuegos "
    "de Steam mediante K-Means"
)

# =====================================
# FILTROS
# =====================================

st.sidebar.header("Filtros")

generos = sorted({
    genero.strip()
    for valor in df["genres"].dropna()
    for genero in str(valor).split(";")
    if genero.strip()
})

genero = st.sidebar.selectbox(
    "Género",
    ["Todos"] + generos
)

anios = sorted(
    df["Anio_Lanzamiento"].dropna().astype(int).unique()
)

anio = st.sidebar.selectbox(
    "Año de lanzamiento",
    ["Todos"] + list(anios)
)

grupo = st.sidebar.selectbox(
    "Clúster",
    ["Todos", 0, 1]
)

filtrado = df.copy()

if genero != "Todos":
    filtrado = filtrado[
        filtrado["genres"].fillna("").str.split(";").apply(
            lambda lista: genero in [
                elemento.strip() for elemento in lista
            ]
        )
    ]

if anio != "Todos":
    filtrado = filtrado[
        filtrado["Anio_Lanzamiento"] == anio
    ]

if grupo != "Todos":
    filtrado = filtrado[
        filtrado["Grupo"] == grupo
    ]

st.sidebar.caption(
    f"Mostrando {len(filtrado):,} de {len(df):,} videojuegos"
)

pagina = st.sidebar.radio(
    "Navegación",
    ["Vista general", "Segmentación K-Means"]
)

# =====================================
# INDICADORES KPI
# =====================================

total = len(filtrado)
precio = filtrado["price"].mean()
resenas = filtrado["Total_Resenas"].sum()
propietarios = filtrado["Owners_Estimados"].sum()

k1, k2, k3, k4 = st.columns(4)

k1.metric("Videojuegos", f"{total:,}")
k2.metric(
    "Precio promedio",
    f"{precio:.2f}" if pd.notna(precio) else "N/D"
)
k3.metric("Reseñas registradas", f"{resenas:,.0f}")
k4.metric(
    "Propietarios estimados",
    f"{propietarios:,.0f}"
)

st.divider()

# =====================================
# PAGINA 1: VISTA GENERAL
# =====================================

if pagina == "Vista general":

    st.header("Panorama general de Steam")

    izquierda, derecha = st.columns(2)

    with izquierda:
        st.subheader("Distribución por clúster")

        distribucion = (
            filtrado["Segmento"]
            .value_counts()
            .reset_index()
        )
        distribucion.columns = ["Segmento", "Cantidad"]

        if not distribucion.empty:
            fig = px.pie(
                distribucion,
                names="Segmento",
                values="Cantidad",
                hole=0.45
            )
            st.plotly_chart(
                fig, use_container_width=True
            )
        else:
            st.info("No hay datos para estos filtros.")

    with derecha:
        st.subheader("Videojuegos por año")

        por_anio = (
            filtrado.dropna(subset=["Anio_Lanzamiento"])
            .groupby("Anio_Lanzamiento")
            .size()
            .reset_index(name="Cantidad")
        )

        fig = px.bar(
            por_anio,
            x="Anio_Lanzamiento",
            y="Cantidad",
            labels={
                "Anio_Lanzamiento": "Año",
                "Cantidad": "Videojuegos"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Géneros más frecuentes")

    tabla_generos = (
        filtrado["genres"]
        .dropna()
        .str.split(";")
        .explode()
        .str.strip()
        .value_counts()
        .head(10)
        .reset_index()
    )
    tabla_generos.columns = ["Género", "Cantidad"]

    fig = px.bar(
        tabla_generos,
        x="Cantidad",
        y="Género",
        orientation="h"
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Catálogo de videojuegos")

    columnas = [
        "name",
        "Anio_Lanzamiento",
        "genres",
        "price",
        "Owners_Estimados",
        "Total_Resenas",
        "Porcentaje_Positivo",
        "Grupo"
    ]

    st.dataframe(
        filtrado[columnas],
        use_container_width=True,
        hide_index=True
    )

# =====================================
# PAGINA 2: SEGMENTACION
# =====================================

else:

    st.header("Segmentación mediante K-Means")

    st.markdown("""
    **Grupo 0 — Baja actividad registrada:** títulos con
    menor cantidad de propietarios, reseñas y tiempo
    promedio de juego registrado.

    **Grupo 1 — Alta participación:** títulos con mayor
    alcance estimado, más reseñas y tiempos de juego
    registrados superiores.
    """)

    st.subheader("Distribución de los grupos")

    distribucion = (
        filtrado.groupby("Grupo")
        .size()
        .reset_index(name="Videojuegos")
    )
    distribucion["Grupo"] = distribucion["Grupo"].astype(str)

    fig = px.bar(
        distribucion,
        x="Grupo",
        y="Videojuegos",
        color="Grupo",
        text="Videojuegos"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dispersión de videojuegos")

    # Se muestra una proyección de dos variables
    # del modelo original de cinco dimensiones.
    fig = px.scatter(
        filtrado,
        x="price",
        y="Total_Resenas",
        color="Segmento",
        hover_name="name",
        hover_data=[
            "Owners_Estimados",
            "Porcentaje_Positivo",
            "average_playtime"
        ],
        log_y=True,
        opacity=0.45,
        labels={
            "price": "Precio",
            "Total_Resenas": "Total de reseñas"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "La dispersión muestra dos variables originales. "
        "K-Means utilizó cinco variables transformadas "
        "y normalizadas."
    )

    st.subheader("Comparación de estadísticas")

    variable = st.selectbox(
        "Variable a comparar",
        [
            "price",
            "Owners_Estimados",
            "Total_Resenas",
            "Porcentaje_Positivo",
            "average_playtime"
        ]
    )

    estadisticas = (
        filtrado.groupby("Grupo")[variable]
        .agg(["mean", "median"])
        .reset_index()
    )

    estadisticas["Grupo"] = (
        estadisticas["Grupo"].astype(str)
    )

    fig = px.bar(
        estadisticas,
        x="Grupo",
        y=["mean", "median"],
        barmode="group",
        labels={
            "value": "Valor",
            "variable": "Estadístico"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Estadísticas por segmento")
    st.dataframe(
        estadisticas,
        use_container_width=True,
        hide_index=True
    )

    if EVALUACION.exists():
        st.subheader("Evaluación del número de clústeres")
        evaluacion = pd.read_csv(EVALUACION)

        st.dataframe(
            evaluacion,
            use_container_width=True,
            hide_index=True
        )

        fig = px.line(
            evaluacion,
            x="K",
            y="Silueta",
            markers=True,
            title="Coeficiente de silueta por K"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Hallazgos del análisis")

    st.markdown("""
    1. El segmento de alta participación representa
       aproximadamente el 21% del catálogo completo,
       pero presenta muchas más reseñas y propietarios.

    2. El precio promedio es mayor en el segmento de
       alta participación, aunque esto no demuestra
       una relación causal.

    3. Los géneros Indie, Action y Adventure aparecen
       entre los más frecuentes de ambos segmentos.
    """)

    st.subheader("Recomendaciones")

    st.markdown("""
    - Analizar los títulos de mayor participación
      para identificar características comerciales.
    - Comparar precios, géneros y recepción antes
      de tomar decisiones de distribución.
    - Incorporar datos más recientes para mejorar
      la utilidad comercial del análisis.
    """)

    st.info(
        "Los hallazgos anteriores corresponden al "
        "dataset completo. Los filtros modifican las "
        "gráficas y estadísticas, pero no recalculan "
        "los clústeres."
    )

st.divider()
st.caption(
    "GameScope | Proyecto integrador de Minería de Datos | "
    "Fuente: Steam Store Games (Nik Davis)"
)
