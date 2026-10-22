"""Pestaña 5: reserva teórica (Montecarlo, distribuciones, regla certeza → ley, cómo decidir y glosario)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import Supuestos, simular
from core.distribuciones import CATALOGO, congelar, estadisticos
from core.modelo import CERTEZA_MAX_DEFECTO, CERTEZA_MIN_DEFECTO, factor_ley
from . import graficos as g


def render_teoria() -> None:
    st.markdown("Una reserva teórica para entender **qué hace** la plataforma y **por qué**. Sin fórmulas innecesarias: "
                "primero la idea en palabras, luego el detalle técnico.")
    t1, t2, t3, t4, t5 = st.tabs(["🎲 Montecarlo", "📊 Distribuciones", "🪨 Certeza → ley", "🎯 Cómo decidir", "📖 Glosario"])
    with t1:
        _montecarlo()
    with t2:
        _distribuciones()
    with t3:
        _certeza()
    with t4:
        _decidir()
    with t5:
        _glosario()


# ----------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _demo(certeza: int, semilla: int) -> np.ndarray:
    """Una zona de ejemplo para la demo de convergencia (misma regla que la plataforma)."""
    zona = pd.DataFrame([{"Zona": "Zona demo", "TMS": 10000, "Ley base (g/t)": 6.0, "Recuperación (%)": 91.0,
                          "Precio (US$/oz)": 4500.0, "Costo explotación (US$/t)": 60.0, "Certeza geo-operativa (%)": float(certeza)}])
    r = simular({"Demo": zona}, Supuestos(n_iter=20000, semilla=semilla))["Demo"]
    return r["Utilidad (US$)"].to_numpy()


def _montecarlo() -> None:
    st.markdown("### ¿Qué es la simulación de Montecarlo?")
    st.markdown(
        """
**La idea, en una frase:** en vez de calcular *un solo* resultado con *un solo* valor de cada variable, se calcula
**miles de veces**, cada vez con valores distintos sorteados de lo que creemos posible. El conjunto de resultados
muestra **qué tan probable es cada desenlace**.

**Una analogía:** el plan determinístico es como decir «mañana lloverá 5 mm». Montecarlo es decir «hay 70 % de
probabilidad de que llueva algo y 10 % de que pasen los 20 mm». Con lo segundo decides si llevas paraguas.

**Los pasos que sigue la plataforma:**
1. **Modelo:** las fórmulas de tu Excel (onzas, valorizado, costo, utilidad) por zona.
2. **Incertidumbre:** se elige la variable que no conocemos con exactitud (aquí: la *certeza geo-operativa* de cada zona)
   y una **distribución** que describe su rango (70 %–97 %).
3. **Sorteo:** en cada iteración se saca un valor de cada distribución.
4. **Cálculo:** se evalúa el modelo con esos valores y se guarda la utilidad.
5. **Resultados:** tras 10 000 repeticiones se cuentan los casos: la fracción que cumple una meta es su **probabilidad**.
"""
    )
    st.markdown("### Determinístico vs. estocástico")
    st.dataframe(pd.DataFrame({
        "": ["Entradas", "Resultado", "Responde a", "Riesgo", "En la plataforma"],
        "Determinístico": ["Un valor por variable", "Un número", "«¿Cuánto es el plan?»", "No se ve",
                           "El **plan** (barras de la tabla Total)"],
        "Estocástico (Montecarlo)": ["Un rango y una distribución por variable", "Una distribución de resultados",
                                     "«¿Qué tan probable es cumplir el plan?»", "Se mide (P5, probabilidad de pérdida…)",
                                     "La simulación: curvas y probabilidades"],
    }), hide_index=True, width="stretch")

    st.markdown("### Pruébalo: ¿por qué miles de iteraciones?")
    st.caption("Una zona de ejemplo (10 000 t, ley base 6 g/t, recuperación 91 %, precio US$ 4 500/oz, costo US$ 60/t). "
               "La línea es la utilidad media acumulada: al principio salta, y luego se estabiliza.")
    c1, c2 = st.columns(2)
    certeza = c1.slider("Certeza de la zona en la tabla (%)", 70, 97, 88, key="demo_cert")
    n = c2.select_slider("Iteraciones a considerar", [100, 500, 1000, 5000, 10000, 20000], value=1000, key="demo_n")
    x = _demo(certeza, 11)
    div, suf = g.escala(x)
    st.plotly_chart(g.fig_convergencia(x, div, suf), width="stretch", key="demo_conv")
    xs = x[:n]
    se = xs.std(ddof=1) / np.sqrt(n)
    m = st.columns(3)
    m[0].metric(f"Media con {n:,} iteraciones", f"{xs.mean() / div:,.3f} {suf}")
    m[1].metric("Error estándar de la media", f"± {se / div:,.3f} {suf}")
    m[2].metric("Con 20 000 iteraciones", f"{x.mean() / div:,.3f} {suf}")
    st.latex(r"\text{error estándar}=\frac{\sigma}{\sqrt{N}}")
    st.markdown(
        "Para reducir el error a la mitad hay que **cuadruplicar** las iteraciones. Por eso 10 000 es un buen "
        "equilibrio: la media ya es estable y la simulación sigue siendo instantánea."
    )
    with st.expander("Detalles técnicos que usa la plataforma"):
        st.markdown(
            """
- **Semilla:** número con el que arranca el generador aleatorio. Misma semilla = mismos sorteos = mismo resultado. No es el «número de iteración»: con otra semilla salen otros 10 000 sorteos igual de válidos.
- **Muestreo Latin Hypercube:** en lugar de sortear al azar puro, se divide la probabilidad (0–100 %) en tantos tramos como iteraciones y se saca un valor de cada tramo. Cubre mejor los extremos y estabiliza antes el resultado.
- **Números aleatorios comunes:** (si está activo) cada zona sale igual en todos los escenarios en cada iteración. Así las diferencias entre escenarios vienen del escenario, no de la suerte.
- **Independencia:** cada zona de cada escenario se sortea por separado (ρ = 0). Con ρ > 0 las zonas se correlacionan mediante una cópula gaussiana.
- **Ruido de simulación:** una probabilidad p estimada con N iteraciones tiene error estándar √(p(1−p)/N); con 20 000 es ≤ 0,35 puntos y con 100 000, ≤ 0,16.
"""
        )


# ----------------------------------------------------------------------------------------------
def _distribuciones() -> None:
    st.markdown("### Distribuciones de probabilidad: la forma de la incertidumbre")
    st.markdown(
        "Una **distribución** dice qué valores son más o menos probables. Elegir una es traducir en números lo que "
        "el ingeniero sabe: *cuánto puede variar* y *dónde es más probable* que caiga la variable."
    )
    st.markdown("#### Guía rápida para elegir")
    st.dataframe(pd.DataFrame([
        ("Solo conozco los límites", "Uniforme", "Todos los valores igual de probables"),
        ("Conozco mínimo, más probable y máximo", "Triangular", "Simple y conservadora; la opción por defecto"),
        ("Igual, pero confío más en el valor central", "PERT", "Suaviza los extremos"),
        ("Tengo datos y es simétrico", "Normal", "Campana; se trunca para no salir de rango"),
        ("Variable positiva con subidas fuertes (precio, ley, costo)", "Lognormal", "Cola larga a la derecha. Referencia: no se usa en esta versión"),
        ("Porcentaje entre 0 y 100 con forma flexible", "Beta", "Referencia: aún no está en la plataforma"),
        ("Pocos desenlaces con probabilidad (permiso, paro)", "Discreta", "Referencia: aún no está en la plataforma"),
    ], columns=["Si…", "Usa…", "Porque…"]), hide_index=True, width="stretch")

    st.markdown("#### Explorador interactivo")
    nombre = st.selectbox("Distribución", list(CATALOGO), key="th_dist",
                          format_func=lambda n: f"{n}  ✅ en la plataforma" if CATALOGO[n]["soportada"] else f"{n}  📘 referencia")
    info = CATALOGO[nombre]
    izq, der = st.columns([2, 3])
    with izq:
        st.markdown(f"**En una frase:** {info['resumen']}")
        st.markdown(f"**Dicho fácil:** {info['coloquial']}")
        st.markdown(f"**Técnicamente:** {info['tecnica']}")
        for f in info["formulas"]:
            st.latex(f)
        st.markdown(f"**Cuándo usarla:** {info['cuando']}")
        st.markdown(f"**Ojo:** {info['ojo']}")
    with der:
        if nombre == "Discreta":
            _discreta()
        else:
            params = {}
            cols = st.columns(len(info["params"]))
            for c, (clave, etiqueta, mn, mx, df_, paso) in zip(cols, info["params"]):
                params[clave] = c.slider(etiqueta, mn, mx, df_, paso, key=f"th_{nombre}_{clave}")
            if nombre in ("Triangular", "PERT") and not (params["lo"] <= params["moda"] <= params["hi"]):
                st.warning("Debe cumplirse mínimo ≤ más probable ≤ máximo; se ajusta automáticamente para el gráfico.")
                params["moda"] = min(max(params["moda"], params["lo"]), params["hi"])
            if nombre == "Beta":
                params.setdefault("a", 4.0)
            d = congelar(nombre, **params)
            unidad = "Valor (0 a 1)" if nombre == "Beta" else "Variación respecto a la tabla (%)"
            st.plotly_chart(g.fig_distribucion(nombre, d, unidad=unidad), width="stretch", key=f"th_graf_{nombre}")
            e = estadisticos(d)
            m = st.columns(5)
            for c, (k, v) in zip(m, e.items()):
                k = "Desv. est." if k.startswith("Desv") else k
                c.metric(k, f"{v:,.2f}" if nombre == "Beta" else f"{v:+,.1f}")
            st.caption("Zona sombreada = entre P5 y P95: el 90 % de los sorteos cae ahí. "
                       "La curva de abajo (acumulada) responde «¿qué probabilidad hay de quedar por debajo de x?».")

    st.markdown("#### Cómo traducir tu conocimiento a una distribución")
    st.markdown(
        """
- **Con criterio de experto:** pregunta primero por el **mínimo y el máximo realistas**, luego por el **más probable**. Después contrasta: «¿de verdad hay solo 5 % de probabilidad de salir de este rango?». Las personas suelen subestimar la incertidumbre: amplía el rango si dudas.
- **Con datos:** grafica el histograma y compara su forma con las distribuciones de arriba. Si es simétrico, Normal; si tiene cola a la derecha y es positivo, Lognormal.
- **Elegir no es neutro:** para el mismo rango, la Uniforme pone más probabilidad en los extremos que la Triangular y ésta más que la PERT. Prueba varias: si la decisión no cambia, es robusta.
"""
    )
    with st.expander("🛠️ Si quieres añadir una distribución nueva a la plataforma"):
        st.markdown(
            """
1. Añádela al catálogo en `core/distribuciones.py` (texto, fórmulas, parámetros y función `congelar`).
2. Añade su nombre a `DIST_CERTEZA` y una rama en `certeza_simulada` de `core/modelo.py` (función de percentil inverso aplicada a los uniformes del muestreo).
3. Añade los campos de parámetros en `ui/datos.py`.
4. Cubre el caso con una prueba en `tests/` (media ≈ esperada, dentro de rango).
"""
        )


def _discreta() -> None:
    st.caption("Ejemplo: resultado del permiso de ampliación de la mina.")
    fig = go.Figure(go.Bar(x=["Aprobado", "Aprobado con retraso", "Rechazado"], y=[55, 30, 15], width=0.45,
                           marker_color=[g.PALETA[2], g.PALETA[3], g.PALETA[1]], text=["55 %", "30 %", "15 %"],
                           textposition="outside", textfont=dict(color=g.TEXTO_FUERTE)))
    fig.update_yaxes(title_text="Probabilidad (%)", range=[0, 70])
    st.plotly_chart(g._base(fig, 380, "Discreta: cada desenlace con su probabilidad"), width="stretch", key="th_discreta")


# ----------------------------------------------------------------------------------------------
def _certeza() -> None:
    st.markdown("### Cómo la certeza geo-operativa define la ley estimada")
    st.markdown(
        """
La **certeza geo-operativa** expresa cuánta confianza hay en que la ley de una zona se cumpla en la operación, según la
información disponible (sondajes, densidad de muestreo, categoría del recurso: medido, indicado, inferido) y la capacidad
de ejecutarla. A mayor certeza, la ley base se conserva; a menor certeza, la **ley estimada** queda por debajo de la ley base.

**Regla de trabajo del Excel:**
"""
    )
    st.latex(r"\text{ley estimada}=\text{ley base}+\left(\text{ley base}-\frac{\text{ley base}}{\text{certeza}}\right)=\text{ley base}\times\left(2-\frac{1}{\text{certeza}}\right)")
    st.markdown(
        "Con 100 % de certeza la ley estimada es igual a la ley base; con 70 % conserva el 57,1 %; con 50 % la ley estimada es cero. "
        "En la simulación la certeza se sortea entre un mínimo y un máximo (70 %–97 % por defecto), y por eso la ley, las onzas y la "
        "utilidad cambian en cada iteración."
    )
    c1, c2 = st.columns([3, 2])
    cmin = c2.slider("Mínimo del sorteo (%)", 50, 96, int(CERTEZA_MIN_DEFECTO), key="th_cmin")
    cmax = c2.slider("Máximo del sorteo (%)", 51, 100, int(CERTEZA_MAX_DEFECTO), key="th_cmax")
    c2.markdown("**Ejemplo con ley base 7 g/t**")
    filas = []
    for cz in [100, 97, 90, 88, 80, 70, 60, 50]:
        f = float(factor_ley(cz / 100))
        filas.append({"Certeza (%)": cz, "Se conserva (%)": f * 100, "Ley estimada (g/t)": 7 * f})
    c2.dataframe(pd.DataFrame(filas).style.format({"Se conserva (%)": "{:.1f}", "Ley estimada (g/t)": "{:.2f}"}),
                 hide_index=True, width="stretch")
    with c1:
        puntos = [(f"Certeza {c} %", c, g.PALETA[i]) for i, c in enumerate([70, 88, 97])]
        st.plotly_chart(g.fig_ley_certeza(float(cmin), float(cmax), puntos), width="stretch", key="th_ley_cert")
    st.markdown(
        """
**Qué pasa en la simulación:**
- **Variabilidad:** la certeza de cada zona se sortea en el rango; una zona de certeza baja (70 %) puede subir en la simulación y una de certeza alta (97 %) solo puede bajar, porque el máximo del rango es 97 %.
- **Efecto amplificado:** la ley depende de 1/certeza, así que un punto de certeza pesa más en las zonas de certeza baja que en las de certeza alta.
- **La utilidad se mueve más que la ley:** el costo de explotar una tonelada no baja cuando baja la ley, de modo que una zona de margen estrecho puede perder gran parte de su utilidad con una caída moderada de ley.
"""
    )
    st.info("📌 **Próximas versiones:** separar cada zona en recursos *medido / indicado / inferido*, cada uno con su tonelaje y su certeza.")


# ----------------------------------------------------------------------------------------------
def _decidir() -> None:
    st.markdown("### Cómo leer los resultados para decidir")
    st.markdown(
        """
**El plan (determinístico)** es la utilidad que se obtiene con la certeza geo-operativa de la tabla (la ley estimada ya la incluye).
Es un solo número por escenario. La simulación responde lo que el plan no puede: *¿qué tan probable es lograrlo, y cuánto se pierde si no?*

**La vara de éxito.** Por defecto, éxito = que la utilidad simulada **alcance el 100 % del plan**; puedes bajar la exigencia
(p. ej. 90 %) en la barra lateral. La plataforma mide:

| Medida | Qué responde |
|---|---|
| **Probabilidad de éxito** | ¿Con qué probabilidad la utilidad llega al menos a X % del plan? (X se elige en la barra lateral; 100 % por defecto) |
| **Curva de cumplimiento** | Lo mismo, pero para *todos* los X. Muestra cómo cambia el ranking según qué tan exigente sea la meta |
| **Cumplimiento esperado** | Utilidad media simulada ÷ plan. Cuánto del plan se espera capturar |
| **P5 – P95** | Caso adverso y favorable: el 90 % de los resultados cae entre ambos |
| **Probabilidad de pérdida** | Probabilidad de que la utilidad sea negativa |
| **Prob. de ser el mejor** | En qué fracción de las iteraciones ese escenario tiene la mayor utilidad |
| **Confianza X %** | Utilidad que se supera con X % de probabilidad (el percentil 100 − X) |
"""
    )
    st.markdown(
        """
**Cómo usarlas juntas.**
1. Mira la **probabilidad de éxito**: el escenario con la más alta es el que mejor cumple su plan.
2. Contrástala con el **plan**: si el de mayor utilidad planificada no es el de mayor probabilidad, su plan exige una certeza que es menos probable lograr.
3. Revisa la **curva de cumplimiento**: si las curvas no se cruzan, la elección es robusta a la meta elegida.
4. Mira el **riesgo de cola** (P5, probabilidad de pérdida) y el **tornado** de cada escenario para saber qué variable lo mueve.
5. Considera el **ruido**: diferencias menores que el error estándar (≈ 0,35 puntos con 20 000 iteraciones; ≈ 0,16 con 100 000) no son reales.

**Límites que conviene recordar.** El resultado depende de los supuestos (rango y distribución de la certeza geo-operativa).
Es una herramienta para **comparar escenarios y entender el riesgo**, no una predicción exacta. Esta versión evalúa la utilidad de un periodo; el VAN por años llegará en una versión posterior.
"""
    )


# ----------------------------------------------------------------------------------------------
GLOSARIO = [
    ("Determinístico / plan", "Cálculo con un valor fijo por variable, como el Excel (con la certeza geo-operativa de la tabla)."),
    ("Estocástico", "Cálculo con variables sorteadas de una distribución; entrega un rango de resultados con probabilidades."),
    ("Iteración", "Una repetición del cálculo con un juego de valores sorteados."),
    ("Distribución", "Descripción de qué valores son más o menos probables para una variable."),
    ("Percentil (P5, P50, P95)", "Valor bajo el cual queda ese % de los resultados. P50 es la mediana."),
    ("Confianza X %", "Valor que se supera con probabilidad X %. Equivale al percentil 100 − X."),
    ("Curva de excedencia", "Para cada valor, la probabilidad de superarlo."),
    ("Probabilidad de éxito", "Fracción de iteraciones que alcanzan el criterio (p. ej. ≥ 75 % del plan)."),
    ("Cumplimiento esperado", "Utilidad media simulada dividida por la utilidad del plan."),
    ("Certeza geo-operativa", "Confianza en que la ley de una zona se cumpla en la operación, según la información disponible (0–100 %). Es la variable que se sortea."),
    ("Ley base / ley estimada", "La ley estimada es la ley base ajustada por la certeza: ley base × (2 − 1/certeza)."),
    ("Error estándar", "Cuánto se mueve el promedio simulado por azar de la simulación: σ/√N."),
    ("Semilla", "Número inicial del generador aleatorio; misma semilla, mismos resultados."),
    ("Latin Hypercube", "Muestreo que cubre uniformemente todos los tramos de probabilidad."),
    ("Correlación (ρ)", "Grado en que dos variables se mueven juntas (0 = independientes)."),
    ("Tornado / sensibilidad", "Ranking de qué variable de entrada mueve más el resultado (correlación de Spearman)."),
    ("Recurso medido / indicado / inferido", "Categorías de confianza geológica, de mayor a menor (códigos JORC, CIM, CRIRSCO)."),
]


def _glosario() -> None:
    st.markdown("### Glosario")
    st.dataframe(pd.DataFrame(GLOSARIO, columns=["Término", "Significado"]), hide_index=True, width="stretch")
    st.markdown("### Para profundizar")
    st.markdown(
        """
- Metropolis, N. y Ulam, S. (1949). *The Monte Carlo Method*. Journal of the American Statistical Association — el origen del método.
- Vose, D. *Risk Analysis: A Quantitative Guide* — referencia práctica sobre distribuciones y modelado de riesgo.
- Códigos de reporte de recursos y reservas: **JORC**, **CIM Definition Standards**, **CRIRSCO** (definen medido, indicado e inferido).
- Dimitrakopoulos, R. y grupo de McGill — trabajos sobre **planificación minera estocástica**.

*Revisa título, año y edición exactos antes de citar en un documento formal.*
"""
    )
