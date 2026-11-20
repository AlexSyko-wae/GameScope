# 🎮 GameScope — Inteligencia de Videojuegos

## Descripción

GameScope es un proyecto de minería de datos que utiliza el algoritmo K-Means para analizar y segmentar videojuegos de Steam según sus características comerciales y de participación.

Se analizaron 27,075 videojuegos utilizando cinco variables: precio, propietarios estimados, total de reseñas, porcentaje de valoraciones positivas y tiempo promedio de juego.

## Objetivo

Identificar perfiles de videojuegos mediante técnicas de aprendizaje no supervisado y presentar los resultados en un dashboard interactivo.

## Tecnologías utilizadas

- Python
- Pandas y NumPy
- Scikit-learn
- Matplotlib
- Plotly
- Streamlit

## Resultados

Se evaluaron cinco configuraciones de K-Means (K=2 a K=6).

El modelo seleccionado fue K=2, con un coeficiente de silueta de 0.4449.

| Segmento | Videojuegos | Porcentaje |
|---|---:|---:|
| Grupo 0 — Baja actividad registrada | 21,371 | 78.93% |
| Grupo 1 — Alta participación | 5,704 | 21.07% |

## Ejecución

Instalar dependencias:

```bash
python -m pip install -r requirements.txt
```

Ejecutar los scripts desde la carpeta principal:

```bash
python src/exploracion.py
python src/limpieza.py
python src/clustering.py
python src/interpretacion.py
```

Iniciar el dashboard:

```bash
python -m streamlit run dashboard/app.py
```

## Dataset

Steam Store Games, publicado por Nik Davis en Kaggle.

https://www.kaggle.com/datasets/nikdavis/steam-store-games

## Proyecto académico

**Materia:** Extracción de conocimientos en base de datos

**Grupo:** TI 9-1

**Institución:** UTSLRC

**Integrantes:**
- Ramón Alexandro Vargas Antolín
- Luis Ramón Pérez Álvarez
- Ismael Salazar Negrete

**Docente:** Julia Elizabeth García Herrera

**Fecha:** 23 de noviembre de 2026