import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="GameScope", page_icon="🎮", layout="wide")

BASE = Path(__file__).resolve().parent.parent
ARCHIVO = BASE / "data" / "processed" / "games_clusters.csv"
EVALUACION = BASE / "data" / "processed" / "evaluacion_k.csv"
NOMBRES = {0: "Baja actividad registrada", 1: "Alta participación"}
COLORES = {"Baja actividad registrada": "#4C78A8", "Alta participación": "#F28E2B"}
VARIABLES = {
    "Precio": "price",
    "Propietarios estimados": "Owners_Estimados",
    "Total de reseñas": "Total_Resenas",
    "Valoraciones positivas (%)": "Porcentaje_Positivo",
    "Tiempo promedio de juego (min)": "average_playtime",
}

# Ajustes discretos: se conserva la estructura original y el tema de Streamlit.
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 1.5rem; max-width: 1450px;}
h1, h2, h3 {letter-spacing: -.02em;}
[data-testid="stMetric"] {padding: .9rem 1rem; border: 1px solid rgba(128,128,128,.20); border-radius: 12px;}
[data-testid="stMetricLabel"] {font-weight: 600;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def cargar_datos():
    datos = pd.read_csv(ARCHIVO)
    datos["release_date"] = pd.to_datetime(datos["release_date"], errors="coerce")
    datos["Anio_Lanzamiento"] = datos["release_date"].dt.year
    datos["Segmento"] = datos["Grupo"].map(NOMBRES).fillna("Grupo desconocido")
    return datos


@st.cache_data
def cargar_evaluacion():
    return pd.read_csv(EVALUACION) if EVALUACION.exists() else None


if not ARCHIVO.exists():
    st.error("No se encontró data/processed/games_clusters.csv. Ejecuta primero el análisis del proyecto.")
    st.stop()

df = cargar_datos()
st.title("🎮 GameScope")
st.caption("Sistema de análisis y segmentación de videojuegos de Steam mediante K-Means")

# FILTROS: los mismos tres filtros del diseño original.
st.sidebar.header("Filtros")
generos = sorted({g.strip() for valor in df["genres"].dropna() for g in str(valor).split(";") if g.strip()})
genero = st.sidebar.selectbox("Género", ["Todos"] + generos)
anios = sorted(df["Anio_Lanzamiento"].dropna().astype(int).unique().tolist())
anio = st.sidebar.selectbox("Año de lanzamiento", ["Todos"] + anios)
opciones_grupo = {"Todos": None, "Grupo 0 — Baja actividad": 0, "Grupo 1 — Alta participación": 1}
grupo_etiqueta = st.sidebar.selectbox("Clúster", list(opciones_grupo))
grupo = opciones_grupo[grupo_etiqueta]

filtrado = df.copy()
if genero != "Todos":
    filtrado = filtrado[filtrado["genres"].fillna("").str.split(";").apply(lambda lista: genero in [g.strip() for g in lista])]
if anio != "Todos":
    filtrado = filtrado[filtrado["Anio_Lanzamiento"] == anio]
if grupo is not None:
    filtrado = filtrado[filtrado["Grupo"] == grupo]

st.sidebar.caption(f"Mostrando {len(filtrado):,} de {len(df):,} videojuegos")
st.sidebar.divider()
pagina = st.sidebar.radio("Navegación", ["Vista general", "Segmentación K-Means"])
st.sidebar.caption("Los filtros exploran resultados existentes; no vuelven a entrenar el modelo.")

# INDICADORES KPI
k1, k2, k3, k4 = st.columns(4)
k1.metric("Videojuegos", f"{len(filtrado):,}")
precio = filtrado["price"].mean()
k2.metric("Precio promedio (USD)", f"${precio:,.2f}" if pd.notna(precio) else "N/D")
k3.metric("Reseñas registradas", f"{filtrado['Total_Resenas'].sum():,.0f}")
k4.metric("Propietarios estimados*", f"{filtrado['Owners_Estimados'].sum():,.0f}")
st.caption("* Los propietarios se estiman a partir del punto medio de los rangos del dataset; no son cifras exactas.")
st.divider()

if filtrado.empty:
    st.warning("No hay videojuegos que coincidan con estos filtros. Prueba otra combinación.")
    st.stop()

if pagina == "Vista general":
    st.header("Panorama general de Steam")
    st.caption("Explora la distribución del catálogo por segmento, año y género.")
    izquierda, derecha = st.columns(2)

    with izquierda:
        st.subheader("Distribución por clúster")
        distribucion = filtrado["Segmento"].value_counts().rename_axis("Segmento").reset_index(name="Cantidad")
        fig = px.pie(distribucion, names="Segmento", values="Cantidad", hole=0.45,
                     color="Segmento", color_discrete_map=COLORES)
        fig.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value:,} juegos (%{percent})<extra></extra>")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="Segmento")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Porcentaje de videojuegos que pertenecen a cada grupo dentro de los filtros seleccionados.")

    with derecha:
        st.subheader("Videojuegos por año")
        por_anio = (filtrado.dropna(subset=["Anio_Lanzamiento"])
                    .groupby("Anio_Lanzamiento").size().reset_index(name="Cantidad"))
        if not por_anio.empty:
            por_anio["Anio_Lanzamiento"] = por_anio["Anio_Lanzamiento"].astype(int)
            fig = px.bar(por_anio, x="Anio_Lanzamiento", y="Cantidad",
                         labels={"Anio_Lanzamiento": "Año", "Cantidad": "Videojuegos"})
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(dtick=1 if len(por_anio) <= 12 else 5))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay fechas válidas para mostrar.")
        st.caption("Cantidad de títulos registrados según su año de lanzamiento.")

    st.subheader("Géneros más frecuentes")
    tabla_generos = (filtrado["genres"].dropna().str.split(";").explode().str.strip()
                     .loc[lambda s: s.ne("")].value_counts().head(10).rename_axis("Género").reset_index(name="Cantidad"))
    if not tabla_generos.empty:
        fig = px.bar(tabla_generos, x="Cantidad", y="Género", orientation="h", text="Cantidad")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=10, b=10))
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Un videojuego puede pertenecer a varios géneros, por lo que las cantidades pueden superponerse.")

    st.subheader("Catálogo de videojuegos")
    columnas = {"name": "Videojuego", "Anio_Lanzamiento": "Año", "genres": "Géneros",
                "price": "Precio (USD)", "Owners_Estimados": "Propietarios estimados",
                "Total_Resenas": "Reseñas", "Porcentaje_Positivo": "Positivas (%)", "Segmento": "Segmento"}
    st.dataframe(filtrado[list(columnas)].rename(columns=columnas), use_container_width=True,
                 hide_index=True, column_config={"Precio (USD)": st.column_config.NumberColumn(format="$%.2f"),
                                                  "Positivas (%)": st.column_config.NumberColumn(format="%.1f%%")})

else:
    st.header("Segmentación mediante K-Means")
    st.caption("El modelo agrupa videojuegos según cinco variables: precio, propietarios, reseñas, valoración positiva y tiempo de juego.")
    st.markdown("**Grupo 0 — Baja actividad registrada:** títulos con menos propietarios, reseñas y tiempo de juego registrado.  \n"
                "**Grupo 1 — Alta participación:** títulos con mayor alcance estimado, más reseñas y más tiempo de juego registrado.")

    st.subheader("Distribución de los grupos")
    distribucion = filtrado.groupby(["Grupo", "Segmento"]).size().reset_index(name="Videojuegos")
    distribucion["Grupo"] = distribucion["Grupo"].astype(str)
    fig = px.bar(distribucion, x="Grupo", y="Videojuegos", color="Segmento", text="Videojuegos",
                 color_discrete_map=COLORES, category_orders={"Grupo": ["0", "1"]})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_type="category", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Los grupos se asignaron con el dataset completo. Los filtros solo modifican los videojuegos visibles.")

    st.subheader("Dispersión de videojuegos")
    # Plotly requiere valores estrictamente positivos para usar escala logarítmica.
    puntos = filtrado[filtrado["Total_Resenas"] > 0]
    if not puntos.empty:
        fig = px.scatter(puntos, x="price", y="Total_Resenas", color="Segmento", hover_name="name",
                         hover_data=["Owners_Estimados", "Porcentaje_Positivo", "average_playtime"],
                         log_y=True, opacity=0.45, color_discrete_map=COLORES,
                         labels={"price": "Precio (USD)", "Total_Resenas": "Total de reseñas"})
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay videojuegos con reseñas positivas en cantidad para representar en escala logarítmica.")
    st.caption("Cada punto es un videojuego. El eje de reseñas es logarítmico y omite los títulos con cero reseñas. "
               "Esta gráfica muestra solo dos de las cinco variables utilizadas por K-Means.")

    st.subheader("Comparación de estadísticas")
    nombre_variable = st.selectbox("Variable a comparar", list(VARIABLES))
    variable = VARIABLES[nombre_variable]
    estadisticas = filtrado.groupby("Grupo")[variable].agg(Promedio="mean", Mediana="median").reset_index()
    estadisticas["Grupo"] = estadisticas["Grupo"].astype(str)
    comparacion = estadisticas.melt(id_vars="Grupo", value_vars=["Promedio", "Mediana"],
                                    var_name="Estadístico", value_name="Valor")
    fig = px.bar(comparacion, x="Grupo", y="Valor", color="Estadístico", barmode="group",
                 labels={"Valor": nombre_variable}, category_orders={"Grupo": ["0", "1"]})
    fig.update_layout(xaxis_type="category", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Promedio: media de los valores. Mediana: valor central; suele ser menos sensible a valores extremos.")
    st.subheader("Estadísticas por segmento")
    st.dataframe(estadisticas.rename(columns={"Grupo": "Clúster"}).round(2), use_container_width=True, hide_index=True)

    evaluacion = cargar_evaluacion()
    if evaluacion is not None and {"K", "Silueta"}.issubset(evaluacion.columns):
        st.subheader("Evaluación del número de clústeres")
        st.caption("Se compararon diferentes valores de K usando el coeficiente de silueta.")
        fig = px.line(evaluacion, x="K", y="Silueta", markers=True, title="Coeficiente de silueta por K")
        fig.update_xaxes(dtick=1)
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(evaluacion, use_container_width=True, hide_index=True)
        mejor = evaluacion.loc[evaluacion["Silueta"].idxmax()]
        st.success(f"**K = {int(mejor['K'])}** obtuvo la mayor silueta evaluada ({mejor['Silueta']:.4f}). "
                   "Una silueta más alta indica mejor separación relativa entre grupos.")
    elif evaluacion is None:
        st.info("No se encontró el archivo de evaluación de K.")
    else:
        st.warning("El archivo de evaluación no contiene las columnas K y Silueta.")

    st.subheader("Hallazgos del análisis")
    st.markdown("""
1. En el catálogo completo, el segmento de alta participación reúne aproximadamente el **21 %** de los videojuegos, pero registra más reseñas y propietarios.
2. El precio promedio es mayor en ese segmento; **esto no demuestra causalidad**.
3. **Indie, Action y Adventure** aparecen entre los géneros más frecuentes de ambos segmentos.
""")
    st.subheader("Recomendaciones")
    st.markdown("""
- Examinar los títulos de alta participación para investigar características comerciales.
- Comparar géneros, precios y recepción antes de tomar decisiones.
- Incorporar datos más recientes para mejorar la utilidad del análisis.
""")
    st.info("Los hallazgos anteriores describen el dataset completo. Los filtros modifican las gráficas y estadísticas, "
            "pero no recalculan los clústeres ni cambian el modelo entrenado.")

st.divider()
st.caption("GameScope | Proyecto integrador de Minería de Datos | Fuente: Steam Store Games (Nik Davis)")
