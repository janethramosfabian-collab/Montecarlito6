"""Pestañas 1 y 2: ingreso de escenarios (zonas) y supuestos de incertidumbre."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import Supuestos, limpiar_tabla, tabla_determinista, totales_deterministas, validar_tabla
from core.ejemplo import escenarios_ejemplo, tabla_vacia
from core.distribuciones import CATALOGO, congelar
from core.modelo import (
    CERTEZA_MAX_DEFECTO, CERTEZA_MIN_DEFECTO, COL_CERTEZA, COL_COSTO, COL_LEY, COL_LEY_EST, COL_PRECIO, COL_REC,
    COL_TMS, COL_ZONA, DIST_CERTEZA, ITERACIONES, factor_ley,
)
from . import graficos as g


# ------------------------------------------------------------------ estado
def _nuevo_id() -> int:
    st.session_state["_next_id"] = st.session_state.get("_next_id", 0) + 1
    return st.session_state["_next_id"]


def cargar_ejemplo() -> None:
    st.session_state["esc"] = [
        {"id": _nuevo_id(), "nombre": n, "base": df.copy(), "ver": 0} for n, df in escenarios_ejemplo().items()
    ]
    st.session_state.pop("sim", None)


def init_estado() -> None:
    if "esc" not in st.session_state:
        cargar_ejemplo()


def _agregar(copiar: bool) -> None:
    esc = st.session_state["esc"]
    base = esc[-1]["base"].copy() if (copiar and esc) else tabla_vacia()
    esc.append({"id": _nuevo_id(), "nombre": f"Escenario {len(esc) + 1}", "base": base, "ver": 0})


def _eliminar(i: int) -> None:
    if len(st.session_state["esc"]) > 1:
        st.session_state["esc"].pop(i)


# ------------------------------------------------------------------ pestaña 1
_CONFIG = {
    COL_ZONA: st.column_config.TextColumn("Zona", required=True),
    COL_TMS: st.column_config.NumberColumn("TMS (t)", min_value=0, format="%.0f"),
    COL_LEY: st.column_config.NumberColumn("Ley base (g-Au/t)", min_value=0, format="%.2f"),
    COL_REC: st.column_config.NumberColumn("Recuperación (%)", min_value=0, max_value=100, format="%.1f"),
    COL_PRECIO: st.column_config.NumberColumn("Precio (US$/oz)", min_value=0, format="%.0f"),
    COL_COSTO: st.column_config.NumberColumn("Costo explotación (US$/t)", min_value=0, format="%.1f"),
    COL_CERTEZA: st.column_config.NumberColumn("Certeza geo-operativa (%)", min_value=1, max_value=100, format="%.0f"),
}


def tabla_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Cálculo determinístico con Excel: conceptos en filas, zonas en columnas y columna Total."""
    det = tabla_determinista(df)
    tot = totales_deterministas(df)
    zonas = det["Zona"].astype(str).tolist()

    def fila(concepto, unidad, col, fmt, total, escala=1.0):
        vals = [fmt.format(v * escala) for v in det[col]] if col else [""] * len(zonas)
        return {"Concepto": concepto, "Unid": unidad, **dict(zip(zonas, vals)), "Total": fmt.format(total * escala) if total is not None else ""}

    filas = [
        {"Concepto": "TMS por día (TMS ÷ 30)", "Unid": "t/día", **{z: f"{v / 30:,.0f}" for z, v in zip(zonas, det["TMS"])},
         "Total": f"{tot['TMS'] / 30:,.0f}"},
        fila("TMS", "t", "TMS", "{:,.0f}", tot["TMS"]),
        fila("Ley estimada", "g-Au/t", COL_LEY_EST, "{:,.2f}", tot[COL_LEY_EST]),
        fila("Recuperación", "%", "Recuperación (%)", "{:,.0f}%", tot["Recuperación (%)"]),
        fila("Onzas", "Oz", "Onzas (oz)", "{:,.0f}", tot["Onzas (oz)"]),
        fila("Precio", "$", "Precio (US$/oz)", "{:,.0f}", tot["Precio (US$/oz)"]),
        fila("Valorizado", "$", "Valorizado (US$)", "{:,.0f}", tot["Valorizado (US$)"]),
        fila("Costo Explotación", "$/t", "Costo explotación (US$/t)", "{:,.0f}", None),
        fila("Costo Total", "$", "Costo total (US$)", "{:,.0f}", tot["Costo total (US$)"]),
        fila("UTILIDAD", "$", "Utilidad (US$)", "{:,.0f}", tot["Utilidad (US$)"]),
        fila("Certeza Geo-Operativa", "%", COL_CERTEZA, "{:,.0f}%", tot[COL_CERTEZA]),
        fila("Ley base", "g-Au/t", COL_LEY, "{:,.1f}", tot[COL_LEY]),
        fila("Ley estimada", "g-Au/t", COL_LEY_EST, "{:,.1f}", tot[COL_LEY_EST]),
    ]
    return pd.DataFrame(filas)


def render_escenarios() -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Dibuja las tablas editables y devuelve ({nombre: tabla limpia}, errores)."""
    st.markdown(
        "Ingresa **TMS, ley base, recuperación, precio, costo de explotación y certeza geo-operativa** de cada zona. "
        "La **ley estimada** (`ley base × (2 − 1/certeza)`), las onzas, el valorizado, el costo total y la utilidad "
        "se calculan solos con las fórmulas."
    )
    esc = st.session_state["esc"]
    entradas: dict[str, pd.DataFrame] = {}
    errores: list[str] = []
    vistos: set[str] = set()

    tabs = st.tabs([e["nombre"] for e in esc])
    for i, (tab, e) in enumerate(zip(tabs, esc)):
        with tab:
            c1, c2 = st.columns([5, 3], vertical_alignment="bottom")
            nombre = c1.text_input("Nombre del escenario", value=e["nombre"], key=f"nom_{e['id']}")
            e["nombre"] = nombre.strip() or f"Escenario {i + 1}"
            c2.button("🗑️ Eliminar escenario", key=f"del_{e['id']}", on_click=_eliminar, args=(i,),
                      disabled=len(esc) == 1, width="stretch")

            editado = st.data_editor(
                e["base"], column_config=_CONFIG, num_rows="dynamic", hide_index=True,
                width="stretch", key=f"ed_{e['id']}_{e['ver']}",
            )
            limpio = limpiar_tabla(editado)
            if e["nombre"] in vistos:
                errores.append(f"Hay dos escenarios llamados «{e['nombre']}»: cambia uno de los nombres.")
            vistos.add(e["nombre"])
            errs = validar_tabla(limpio, e["nombre"])
            errores += errs
            if errs:
                for m in errs:
                    st.error(m)
                continue
            entradas[e["nombre"]] = limpio

            st.markdown("**Cálculo determinístico**; la columna Total pondera con las TMS de este escenario)")
            st.dataframe(tabla_excel(limpio), hide_index=True, width="stretch")

    b1, b2, b3 = st.columns([2, 2, 3])
    b1.button("➕ Añadir escenario (copia del último)", on_click=_agregar, args=(True,), width="stretch")
    b2.button("➕ Añadir escenario vacío", on_click=_agregar, args=(False,), width="stretch")
    b3.button("↺ Volver a los 3 escenarios del Excel", on_click=cargar_ejemplo, width="stretch")
    if len(entradas) > 6:
        st.warning("Con más de 6 escenarios los colores de los gráficos se vuelven difíciles de distinguir.")
    return entradas, errores


# ------------------------------------------------------------------ pestaña 2
def _ayuda_distribucion(nombre: str, params: dict, clave: str, unidad: str) -> None:
    """Mini «reserva teórica» bajo el selector: qué es la distribución, cuándo usarla y su forma con tus parámetros."""
    info = CATALOGO[nombre]
    with st.expander(f"📖 ¿Qué es la distribución {nombre}?"):
        st.markdown(f"**En una frase:** {info['resumen']}")
        st.markdown(f"**Dicho fácil:** {info['coloquial']}")
        st.markdown(f"**Cuándo usarla:** {info['cuando']}")
        st.markdown(f"**Ojo:** {info['ojo']}")
        try:
            d = congelar(nombre, **params)
            st.plotly_chart(g.fig_distribucion(nombre, d, unidad=unidad), width="stretch", key=f"dist_{clave}")
            st.caption(f"Con tus parámetros: media {d.mean():.1f} %, P5 {d.ppf(.05):.1f} %, P95 {d.ppf(.95):.1f} %. "
                       "Más detalle y las fórmulas en la pestaña **5 · Teoría**.")
        except Exception:  # noqa: BLE001
            st.caption("Ajusta los parámetros para ver el gráfico.")


def render_supuestos(entradas: dict[str, pd.DataFrame]) -> Supuestos:
    st.markdown(
        "En **cada iteración** de Montecarlo se sortea **una sola cosa: la certeza geo-operativa** de cada zona, entre un "
        "mínimo y un máximo. La certeza sorteada cambia la **ley estimada** (`ley base × (2 − 1/certeza)`) y, con ella, "
        "las onzas y la utilidad. El precio, las TMS, la ley base, la recuperación y los costos quedan fijos."
    )
    s = Supuestos()
    col_p, col_c, col_s = st.columns(3)
    certezas_tabla = [float(c) for df in entradas.values() for c in df[COL_CERTEZA]] if entradas else []
    ref = float(np.mean(certezas_tabla)) if certezas_tabla else 85.0

    with col_p:
        with st.container(border=True):
            st.subheader("📌 Datos fijos")
            st.markdown(
                "Se mantienen iguales en todas las iteraciones:\n"
                "- **Precio** de cada zona (tabla de la pestaña 1)\n"
                "- **TMS** por zona y escenario\n"
                "- **Ley base** por zona y escenario\n"
                "- **Recuperación** y **costo de explotación**"
            )
            if entradas:
                p_ref = float(pd.concat(entradas.values())[COL_PRECIO].mean())
                st.metric("Precio fijo (promedio de las tablas)", f"US$ {p_ref:,.0f}/oz")
            st.caption("Lo único que cambia entre iteraciones es la certeza geo-operativa.")

    with col_c:
        with st.container(border=True):
            st.subheader("🪨 Certeza geo-operativa")
            s.certeza_dist = st.selectbox(
                "Distribución del sorteo", DIST_CERTEZA, index=0, key="c_dist",
                help="Triangular y PERT: el valor más probable de cada zona es su certeza de la tabla. "
                     "Uniforme: todos los valores del rango igual de probables. Normal: campana centrada en la certeza de la tabla.")
            c1, c2 = st.columns(2)
            s.certeza_min_pct = c1.number_input("Mínimo (%)", min_value=1.0, max_value=100.0, value=CERTEZA_MIN_DEFECTO,
                                                step=1.0, key="c_min")
            s.certeza_max_pct = c2.number_input("Máximo (%)", min_value=1.0, max_value=100.0, value=CERTEZA_MAX_DEFECTO,
                                                step=1.0, key="c_max")
            if s.certeza_dist == "Normal":
                s.certeza_sd_pp = st.slider("Desviación estándar (puntos %)", 1.0, 20.0, 7.0, 0.5, key="c_sd")
            st.caption("El rango es el mismo para todas las zonas y escenarios.")
            pc = {"lo": s.certeza_min_pct, "moda": min(max(ref, s.certeza_min_pct), s.certeza_max_pct),
                  "hi": s.certeza_max_pct, "mu": ref, "sd": s.certeza_sd_pp, "cv": 10.0}
            _ayuda_distribucion(s.certeza_dist, pc, "certeza", "Certeza geo-operativa (%)")
            with st.expander("Opciones avanzadas"):
                s.rho_zonas = st.slider("Correlación entre la certeza de las zonas (ρ)", 0.0, 0.95, 0.0, 0.05, key="c_rho",
                                        help="0 = cada zona se sortea de forma independiente. Si subes ρ, "
                                             "cuando una zona sale mala las demás tienden a salir malas también.")
            st.caption("⚠️ El rango y la distribución son supuestos de trabajo: conviene validarlos con el ingeniero.")

    with col_s:
        with st.container(border=True):
            st.subheader("⚙️ Simulación")
            s.n_iter = st.select_slider("Iteraciones", ITERACIONES, value=ITERACIONES[0], key="n_it",
                                        format_func=lambda v: f"{v:,}")
            s.semilla = int(st.number_input("Semilla (para repetir resultados)", value=42, step=1, key="semilla"))
            s.compartir_geologia = st.checkbox(
                "Misma zona = mismo sorteo en todos los escenarios", value=True, key="compartir",
                help="Recomendado: la Zona 1 del Escenario 1 y la del Escenario 2 son la misma roca, así que "
                     "salen iguales en cada iteración y la comparación entre escenarios es justa.")

    with st.container(border=True):
        st.markdown("#### 📉 Así afecta la certeza a la ley")
        a, b = st.columns([3, 2])
        puntos = []
        for i, (n, df) in enumerate(entradas.items()):
            for _, z in df.iterrows():
                puntos.append((f"{n} · {z[COL_ZONA]}", float(z[COL_CERTEZA]), g.color_de(i)))
        with a:
            st.plotly_chart(g.fig_ley_certeza(s.certeza_min_pct, s.certeza_max_pct, puntos), width="stretch",
                            key="graf_ley_certeza")
            st.caption("Cada punto es una zona de un escenario (color = escenario) con la certeza de la tabla. "
                       "La banda sombreada es el rango en que se sortea la certeza: cuanto más a la izquierda, menos ley.")
        with b:
            st.markdown("**🧮 Calculadora**")
            ley = st.number_input("Ley base (g/t)", value=7.0, min_value=0.0, step=0.5, key="calc_ley")
            cz = st.slider("Certeza geo-operativa (%)", 50, 100, 70, key="calc_cz")
            f = float(factor_ley(cz / 100))
            st.metric("Ley estimada", f"{ley * f:,.2f} g/t", f"{(f - 1) * 100:,.1f} % ({ley * (f - 1):+.2f} g/t)")
            st.latex(r"\text{ley estimada}=\text{ley base}+\left(\text{ley base}-\frac{\text{ley base}}{\text{certeza}}\right)")

    with st.expander("¿Cómo se calcula cada iteración?"):
        st.markdown(
            """
1. **Certeza de cada zona:** se sortea entre el mínimo y el máximo (70 %–97 % por defecto) con la distribución elegida. El valor más probable de cada zona es su certeza de la tabla.
2. **Ley estimada:** `ley base × (2 − 1/certeza)`, la misma fórmula del Excel. Menos certeza → menos ley.
3. **Onzas** `= TMS × ley estimada × recuperación / 31.1035`, **valorizado** `= onzas × precio` (fijo), **costo** `= costo/t × TMS` y **utilidad** `= valorizado − costo`.
4. Se repite miles de veces (20 000, 30 000, 50 000 o 100 000). Se compara cada resultado con el **plan** (el determinístico, calculado con la certeza de la tabla) y se calcula la **probabilidad de alcanzar al menos X % del plan**.
"""
        )
    mensajes = []
    if s.certeza_min_pct >= s.certeza_max_pct:
        mensajes.append("Certeza: el mínimo debe ser menor que el máximo.")
    for m in mensajes:
        st.error(m)
    st.session_state["errores_supuestos"] = mensajes
    return s
