# Alokatzaile - Estimador y Comparador de Alquiler en España

Proyecto de Data Science del Bootcamp The Bridge (promoción 2025/2026).

Aplicación web para estimar y comparar precios de alquiler en las 50 provincias españolas, combinando modelos de machine learning con datos socioeconómicos oficiales del INE.

---

## ¿Qué hace?

**Tasador** — estima el precio de alquiler mensual de una vivienda a partir de su superficie y provincia, usando un modelo de regresión lineal entrenado por provincia.

**Comparador** — dado un piso con sus características completas (superficie, habitaciones, baños, terraza, ascensor, piscina, calefacción y provincia), predice cuánto costaría ese mismo piso en las 50 provincias españolas. Si el usuario introduce su precio actual, el sistema diagnostica si está pagando por encima o por debajo del mercado.

---

## Lo que hemos construido

- **EDA completo** del dataset: limpieza, análisis de outliers, distribuciones, correlaciones, análisis por provincia
- **Extracción de variables** desde la columna de texto `features` (terraza, ascensor, piscina, calefacción) mediante expresiones regulares
- **Enriquecimiento con datos del INE**: renta neta per cápita, tasa de paro, precio de compraventa por m² y población por provincia (fuentes: INE Atlas ADRH 2022, EPA Q4 2022, Ministerio de Transportes 2022)
- **Análisis de asequibilidad**: qué porcentaje del salario se destina al alquiler por provincia
- **Segmentación en 3 grupos** por precio mediano provincial para análisis diferenciado
- **Comparativa de modelos**: Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost
- **Modelo enriquecido** con todas las features para el comparador (R² = 0.52, RMSE = 425€)
- **API Flask** con endpoints de tasación y comparación interprovincial
- **Despliegue** en PythonAnywhere

---

## Stack

| Capa       | Tecnología                                |
| ---------- | ------------------------------------------ |
| Análisis  | Python, Pandas, NumPy, Matplotlib, Seaborn |
| Modelos    | scikit-learn, XGBoost                      |
| API        | Flask, Flask-CORS                          |
| Despliegue | PythonAnywhere                             |

---

## Estructura

```
├── notebooks/
│   ├── eda_notebook.ipynb              # EDA completo del dataset
│   ├── split_notebook.ipynb            # Segmentación por grupos de precio
│   ├── model_notebook.ipynb            # Modelo base (regresión lineal + OHE)
│   ├── 02_eda_enriquecido.ipynb        # EDA enriquecido con datos INE
│   ├── 03_modelo_enriquecido.ipynb     # Comparativa de algoritmos ML
│   └── 04_comparativa_provincial.ipynb # Lógica del comparador
│
├── api/
│   ├── app.py                          # API Flask — endpoints /prediction y /compare
│   └── models/
│       ├── lr_Madrid.pkl               # Modelos por provincia (tasador)
│       ├── lr_Barcelona.pkl
│       ├── ...
│       └── model_enriquecido.pkl       # Modelo completo (comparador)
│
└── data/
    ├── housing.csv                     # Dataset original (101.365 registros)
    ├── housing_clean.csv               # Dataset limpio (93.360 registros)
    ├── housing_grupo1.csv              # Provincias precio bajo (<600€ mediana)
    ├── housing_grupo2.csv              # Provincias precio medio (600–875€)
    ├── housing_grupo3.csv              # Provincias precio alto (>875€)
    └── ine_provincias_2022.csv         # Datos socioeconómicos INE 2022
```

---

## Dataset

Dataset original: **101.365 anuncios de alquiler** en 50 provincias españolas (octubre 2022).

Tras limpieza: **93.360 registros** con los siguientes filtros aplicados:

| Variable  | Filtro             |
| --------- | ------------------ |
| price     | 1€ — 5.000€/mes |
| surface   | 1 — 300 m²       |
| bedrooms  | 1 — 10            |
| restrooms | 1 — 5             |

Las propiedades de más de 500 m² (chalets y villas de lujo, 0.46% del dataset) se analizan en el EDA pero se excluyen del modelo base por ser un segmento diferenciado.

---

## Modelos

### Tasador (`/prediction`)

Un modelo de regresión lineal independiente por provincia. Cada modelo se entrena con los datos de alquiler de su propia provincia. Simple, rápido e interpretable.

### Comparador (`/compare`)

Modelo Gradient Boosting único entrenado con las 50 provincias, con las siguientes features:

- Superficie, habitaciones, baños
- Terraza, Ascensor, Piscina, Calefacción (variables binarias extraídas de `features`)
- Provincia (OneHot Encoding)
- Renta neta per cápita provincial (INE ADRH 2022)
- Tasa de paro provincial (INE EPA Q4 2022)

**Comparativa de algoritmos (configuración completa):**

| Modelo            | RMSE  | R²  |
| ----------------- | ----- | ---- |
| Linear Regression | 451€ | 0.47 |
| Ridge             | 447€ | 0.47 |
| Random Forest     | 441€ | 0.49 |
| Gradient Boosting | 425€ | 0.52 |
| XGBoost           | 418€ | 0.54 |

---

## API

Desplegada en PythonAnywhere: `https://varushet.pythonanywhere.com`

### `GET /prediction`

```
GET /prediction?surface=80&provincia=Madrid
```

### `GET /compare`

```
GET /compare?surface=120&provincia=Bizkaia&bedrooms=3&restrooms=2&terraza=1&ascensor=1&precio_publicado=1400
```

| Parámetro       | Tipo   | Default     | Descripción                      |
| ---------------- | ------ | ----------- | --------------------------------- |
| surface          | int    | obligatorio | Superficie en m²                 |
| provincia        | string | obligatorio | Provincia de origen               |
| bedrooms         | int    | 2           | Nº de habitaciones               |
| restrooms        | int    | 1           | Nº de baños                     |
| terraza          | 0/1    | 0           | Tiene terraza                     |
| ascensor         | 0/1    | 0           | Tiene ascensor                    |
| piscina          | 0/1    | 0           | Tiene piscina                     |
| calefaccion      | 0/1    | 0           | Tiene calefacción                |
| precio_publicado | float  | opcional    | Precio actual (para diagnóstico) |

**Respuesta:**

```json
{
  "ranking": [
    { "provincia": "Illes Balears", "precio_predicho": 1680.0 },
    { "provincia": "Barcelona",     "precio_predicho": 1590.0 },
    { "provincia": "Bizkaia",       "precio_predicho": 1320.0 }
  ],
  "ranking_posicion": 3,
  "ranking_total": 50,
  "diagnostico": {
    "precio_publicado": 1400.0,
    "precio_predicho":  1320.0,
    "diferencia_euros": 80.0,
    "diferencia_pct":   6.1,
    "estado": "EN PRECIO DE MERCADO"
  }
}
```

---

## Fuentes de datos

| Dataset                           | Fuente                                                     |
| --------------------------------- | ---------------------------------------------------------- |
| Anuncios de alquiler              | Dataset propio del bootcamp (octubre 2022)                 |
| Renta neta media por persona 2022 | INE — Atlas de Distribución de Renta de los Hogares 2022 |
| Población 2022                   | INE — Padrón Municipal 2022                              |
| Tasa de paro Q4 2022              | INE — Encuesta de Población Activa Q4 2022               |
| Precio m² compraventa 2022       | Ministerio de Transportes, Movilidad y Agenda Urbana       |

---

## Equipo

Danillo Souza, Alejandro Serrato, Oscar Fernández, Urko Menendez y Andoni Olaso - Bootcamp Data Science The Bridge, promoción 2025/2026.
