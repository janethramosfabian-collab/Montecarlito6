"""Catálogo teórico de distribuciones (técnico + coloquial) y utilidades para graficarlas."""
from __future__ import annotations

import numpy as np
from scipy import stats

# Parámetros de cada distribución para el explorador: (clave, etiqueta, mínimo, máximo, defecto, paso)
_P_MIN_MODA_MAX = [
    ("lo", "Mínimo", -60.0, 0.0, -20.0, 1.0),
    ("moda", "Más probable", -30.0, 30.0, 0.0, 1.0),
    ("hi", "Máximo", 0.0, 60.0, 20.0, 1.0),
]

CATALOGO: dict[str, dict] = {
    "Triangular": {
        "soportada": True,
        "resumen": "Tres números: mínimo, más probable y máximo.",
        "coloquial": (
            "Es lo que dice un ingeniero cuando le preguntas «¿entre cuánto y cuánto puede variar?»: "
            "«creo que va entre A y B, y lo más probable es C». La probabilidad sube en línea recta desde el "
            "mínimo hasta el valor más probable y luego baja en línea recta hasta el máximo."
        ),
        "tecnica": (
            "Distribución continua acotada [a, b] con moda c. Su densidad es lineal a tramos, con máximo en c. "
            "Se ajusta con solo tres juicios de experto y no requiere datos históricos."
        ),
        "formulas": [r"\mu=\frac{a+b+c}{3}", r"\sigma^2=\frac{a^2+b^2+c^2-ab-ac-bc}{18}"],
        "cuando": "Pocos datos y buen criterio de experto. Es la opción por defecto para la certeza geo-operativa en esta plataforma.",
        "ojo": "Pone mucho peso en los extremos si el rango es amplio, y la media no coincide con la moda si el rango es asimétrico.",
        "params": _P_MIN_MODA_MAX,
    },
    "PERT": {
        "soportada": True,
        "resumen": "Como la triangular, pero más concentrada en lo más probable.",
        "coloquial": (
            "Es una triangular «suavizada»: usa los mismos tres números (mínimo, más probable, máximo) pero "
            "da más peso a lo más probable y menos a los extremos. Es muy usada en gestión de proyectos "
            "porque refleja mejor la sensación de que lo extremo es raro."
        ),
        "tecnica": (
            "Es una Beta reescalada al rango [a, b] con α = 1 + 4(c−a)/(b−a) y β = 1 + 4(b−c)/(b−a). "
            "Su media pondera 4 veces la moda."
        ),
        "formulas": [r"\mu=\frac{a+4c+b}{6}", r"\sigma\approx\frac{b-a}{6}"],
        "cuando": "Cuando tienes los mismos tres números que la triangular pero confías más en el valor central.",
        "ojo": "Reduce la probabilidad de los extremos: si el riesgo está justamente en las colas, la triangular es más conservadora.",
        "params": _P_MIN_MODA_MAX,
    },
    "Uniforme": {
        "soportada": True,
        "resumen": "Todos los valores del rango son igual de probables.",
        "coloquial": (
            "Es decir «no tengo idea cuál es más probable: puede ser cualquiera entre A y B». "
            "Es la forma honesta de expresar ignorancia dentro de un rango."
        ),
        "tecnica": "Densidad constante 1/(b−a) en [a, b] y cero fuera. Es la distribución de máxima entropía en un rango.",
        "formulas": [r"\mu=\frac{a+b}{2}", r"\sigma^2=\frac{(b-a)^2}{12}"],
        "cuando": "Cuando solo se conocen los límites y no hay razón para preferir ningún valor.",
        "ojo": "Es la que más incertidumbre mete para un mismo rango. Rara vez describe bien un precio o una ley real.",
        "params": [("lo", "Mínimo", -60.0, 0.0, -20.0, 1.0), ("hi", "Máximo", 0.0, 60.0, 20.0, 1.0)],
    },
    "Normal": {
        "soportada": True,
        "resumen": "La campana: valores cerca del promedio son comunes; los lejanos, raros.",
        "coloquial": (
            "Es la clásica campana de Gauss. Sirve cuando el valor es el resultado de muchos pequeños efectos "
            "que se suman (errores de medición, variaciones diarias). Dos tercios de los valores caen a menos de "
            "una desviación estándar del promedio."
        ),
        "tecnica": (
            "Simétrica y no acotada, definida por la media μ y la desviación estándar σ. En esta plataforma se "
            "trunca para evitar precios negativos o certezas fuera de 0–100 %."
        ),
        "formulas": [r"f(x)=\frac{1}{\sigma\sqrt{2\pi}}\,e^{-\frac{(x-\mu)^2}{2\sigma^2}}"],
        "cuando": "Hay datos históricos y el comportamiento es simétrico alrededor de un valor central.",
        "ojo": "Al ser simétrica asigna tanta probabilidad a la subida como a la bajada, y sus colas no tienen límite.",
        "params": [("mu", "Media", -30.0, 30.0, 0.0, 1.0), ("sd", "Desv. estándar", 1.0, 30.0, 10.0, 0.5)],
    },
    "Lognormal": {
        "soportada": False,
        "resumen": "Siempre positiva y con cola larga hacia arriba.",
        "coloquial": (
            "Es la «campana» de las cosas que no pueden ser negativas y que pueden dispararse hacia arriba: "
            "precios, leyes, costos. Lo más común queda un poco por debajo del promedio, pero de vez en cuando "
            "aparece un valor muy alto."
        ),
        "tecnica": (
            "Si ln X es normal, X es lognormal. Se parametriza por su media y su coeficiente de variación "
            "(CV = desviación / media). Es asimétrica a la derecha."
        ),
        "formulas": [r"\ln X\sim N(\mu_L,\sigma_L^2)", r"E[X]=e^{\mu_L+\sigma_L^2/2}", r"CV^2=e^{\sigma_L^2}-1"],
        "cuando": "Variables positivas con variación porcentual (precio de metales, ley, costos).",
        "ojo": "Con CV grande la media se aleja mucho de la moda; y el 50 % de los casos queda por debajo de la media.",
        "params": [("cv", "Coef. de variación (%)", 1.0, 50.0, 10.0, 0.5)],
    },
    "Beta": {
        "soportada": False,
        "resumen": "Muy flexible dentro de un rango; ideal para porcentajes.",
        "coloquial": (
            "Es la «plastilina» de las distribuciones acotadas: según sus dos parámetros puede ser simétrica, "
            "sesgada a un lado, en forma de U o plana. Perfecta para cosas que van de 0 a 100 %, como una "
            "certeza o una recuperación."
        ),
        "tecnica": (
            "Definida en [0, 1] (o reescalada a [a, b]) con parámetros de forma α y β. La PERT es un caso "
            "particular de la Beta."
        ),
        "formulas": [r"f(x)\propto x^{\alpha-1}(1-x)^{\beta-1}", r"\mu=\frac{\alpha}{\alpha+\beta}"],
        "cuando": "Porcentajes o proporciones con datos que permitan estimar su forma (p. ej. muestras de recuperación).",
        "ojo": "Los parámetros α y β no son intuitivos; conviene ajustarlos con datos o con media y desviación.",
        "params": [("a", "α (forma)", 0.5, 15.0, 4.0, 0.5), ("b", "β (forma)", 0.5, 15.0, 2.0, 0.5)],
    },
    "Discreta": {
        "soportada": False,
        "resumen": "Pocos resultados posibles, cada uno con su probabilidad.",
        "coloquial": (
            "Cuando el futuro se reduce a «pasa A con 30 %, pasa B con 50 %, pasa C con 20 %»: aprobar o no un "
            "permiso, que una zona resulte mineralizada o estéril, un paro de operaciones."
        ),
        "tecnica": "Asigna una probabilidad pᵢ a cada valor xᵢ, con Σpᵢ = 1.",
        "formulas": [r"P(X=x_i)=p_i,\quad \sum_i p_i=1"],
        "cuando": "Eventos de riesgo con resultados distintos y bien definidos, no un continuo.",
        "ojo": "Las probabilidades suelen ser subjetivas; documenta de dónde salen.",
        "params": [],
    },
}

SOPORTADAS = [n for n, d in CATALOGO.items() if d["soportada"]]


def congelar(nombre: str, **p):
    """Distribución de scipy para graficar. Valores en «variación %» salvo Beta (0-1) y Discreta."""
    if nombre in ("Triangular",):
        lo, mo, hi = p["lo"], p["moda"], p["hi"]
        mo = min(max(mo, lo), hi)
        return stats.triang(c=(mo - lo) / max(hi - lo, 1e-9), loc=lo, scale=max(hi - lo, 1e-9))
    if nombre == "PERT":
        lo, mo, hi = p["lo"], p["moda"], p["hi"]
        mo = min(max(mo, lo), hi)
        r = max(hi - lo, 1e-9)
        return stats.beta(1 + 4 * (mo - lo) / r, 1 + 4 * (hi - mo) / r, loc=lo, scale=r)
    if nombre == "Uniforme":
        return stats.uniform(loc=p["lo"], scale=max(p["hi"] - p["lo"], 1e-9))
    if nombre == "Normal":
        return stats.norm(p["mu"], max(p["sd"], 1e-9))
    if nombre == "Lognormal":  # factor de media 1; el eje es la variación (%) = (factor − 1)·100
        cv = max(p["cv"] / 100.0, 1e-9)
        s2 = np.log1p(cv**2)
        return stats.lognorm(s=np.sqrt(s2), loc=-100.0, scale=100.0 * np.exp(-s2 / 2))
    if nombre == "Beta":
        return stats.beta(p["a"], p["b"])
    raise ValueError(f"Sin gráfico continuo para {nombre}")


def estadisticos(d) -> dict[str, float]:
    return {
        "Media": float(d.mean()), "Desv. estándar": float(d.std()),
        "P5": float(d.ppf(0.05)), "P50": float(d.ppf(0.50)), "P95": float(d.ppf(0.95)),
    }
