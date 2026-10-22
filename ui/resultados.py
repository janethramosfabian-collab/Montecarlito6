"""Pestaña 3 (resultados por escenario) y criterio de éxito compartido por todas las pestañas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import totales_deterministas
from core.analisis import tabla_zonas
from core.estadistica import COL_U, prob_cumplimiento, resumen, sensibilidad, umbral_escenario
from . import graficos as g

COL_O = "Onzas (oz)"
PCT_DEFECTO = 100


def md(texto: str) -> str:
    """Escapa el signo $ para que Streamlit no lo lea como fórmula."""
    return texto.replace("$", "\\$")


def render_criterio(sim: dict | None) -> None:
    """Criterio de éxito en la barra lateral: vale para Resultados, Comparación y el PDF."""
    st.subheader("Criterio de éxito")
    if sim is None:
        st.caption("Aparece al simular.")
        return
    det_u = {n: totales_deterministas(df)["Utilidad (US$)"] for n, df in sim["entradas"].items()}
    div, suf = g.escala(list(det_u.values()))
    modo_txt = st.radio("Medir el éxito como", ["% del plan de cada escenario", "Monto fijo para todos"],
                        key="u_modo", label_visibility="collapsed")
    if modo_txt.startswith("%"):
        valor = st.slider("Alcanzar al menos (% del plan)", 50, 120, PCT_DEFECTO, 1, key="u_pct",
                          help="Éxito = la utilidad simulada llega a este % de la utilidad planificada (determinístico). "
                               "100 % = cumplir el plan; 90 % = aceptar hasta un 10 % menos.")
        st.session_state["criterio"] = ("pct", float(valor))
    else:
        por_defecto = round(min(det_u.values()) / div, 2)
        v = st.number_input(f"Utilidad mínima (US$ {suf})", value=por_defecto,
                            step=max(round(por_defecto / 50, 2), 0.01), key=f"u_monto_{suf}")
        st.session_state["criterio"] = ("monto", float(v) * div)


def contexto() -> dict | None:
    """Todo lo derivado de la última simulación (plan, umbrales, tabla resumen) para las pestañas de resultados."""
    sim = st.session_state.get("sim")
    if sim is None:
        return None
    res, entradas_s, s = sim["resultados"], sim["entradas"], sim["supuestos"]
    orden = list(res)
    tot = {n: totales_deterministas(entradas_s[n]) for n in orden}
    det_u = {n: tot[n]["Utilidad (US$)"] for n in orden}
    modo, valor = st.session_state.get("criterio", ("pct", float(PCT_DEFECTO)))
    div, suf = g.escala(list(det_u.values()))
    umbrales = {n: umbral_escenario(det_u[n], modo, valor) for n in orden}
    tabla = resumen(res, det_u, modo, valor)
    st.session_state["tabla_resumen"] = tabla
    st.session_state["umbrales"] = umbrales
    return dict(sim=sim, res=res, entradas=entradas_s, s=s, orden=orden, tot=tot, det_u=det_u, modo=modo,
                valor=valor, div=div, suf=suf, umbrales=umbrales, tabla=tabla,
                certeza_pond={n: tot[n]["Certeza geo-operativa (%)"] for n in orden})


def aviso_desactualizado(sim: dict) -> None:
    if sim["firma"] != st.session_state.get("firma_actual"):
        st.warning("Cambiaste datos o supuestos después de la última simulación. Pulsa **▶ Simular** para actualizar; "
                   "abajo se muestran los resultados anteriores.")


# ---------------------------------------------------------------------------------------
def render_resultados(entradas: dict[str, pd.DataFrame]) -> None:
    nombres = list(entradas)
    det = pd.DataFrame({n: totales_deterministas(df) for n, df in entradas.items()})
    st.subheader("1 · Plan determinístico")
    c1, c2 = st.columns([5, 4])
    with c1:
        st.plotly_chart(
            g.fig_determinista(nombres, [det[n]["Utilidad (US$)"] for n in nombres],
                               [det[n]["Onzas (oz)"] for n in nombres]),
            width="stretch", key="graf_det")
    with c2:
        st.markdown("**Totales por escenario** (como la columna *Total* del Excel)")
        st.dataframe(det.style.format("{:,.2f}"), width="stretch")
        st.caption("El plan usa un solo valor por variable (la certeza geo-operativa de la tabla y el precio fijo): "
                   "no dice qué tan probable es cumplirlo. Eso lo responde la simulación, que hace variar la certeza.")

    ctx = contexto()
    st.divider()
    st.subheader("2 · Simulación de Montecarlo, escenario por escenario")
    if ctx is None:
        st.info("Configura los escenarios y los supuestos y pulsa **▶ Simular** en la barra lateral.")
        return
    aviso_desactualizado(ctx["sim"])
    st.caption("Aquí ves cada escenario en detalle. Para ponerlos lado a lado y decidir, usa la pestaña **4 · Comparación**.")
    tabs = st.tabs([f"🔍 {n}" for n in ctx["orden"]])
    for i, (tab, n) in enumerate(zip(tabs, ctx["orden"])):
        with tab:
            _detalle(n, i, ctx)


def _detalle(n: str, i: int, ctx: dict) -> None:
    df, entrada = ctx["res"][n], ctx["entradas"][n]
    det_u, umbral, div, suf, modo, valor = ctx["det_u"][n], ctx["umbrales"][n], ctx["div"], ctx["suf"], ctx["modo"], ctx["valor"]
    x = df[COL_U].to_numpy()
    p5, p95 = np.percentile(x, [5, 95])
    prob = float(np.mean(x >= umbral))
    m = st.columns(6)
    m[0].metric("Plan (determinístico)", f"{det_u / div:,.2f} {suf}")
    m[1].metric("Media simulada", f"{x.mean() / div:,.2f} {suf}")
    m[2].metric("Cumplimiento esperado", f"{x.mean() / det_u:.0%}" if det_u > 0 else "—",
                help="Utilidad media simulada ÷ utilidad del plan.")
    m[3].metric("Prob. de éxito", f"{prob:.1%}",
                help=("Probabilidad de que la utilidad llegue al " + f"{valor:.0f} % del plan." if modo == "pct"
                      else "Probabilidad de superar el monto mínimo."))
    m[4].metric("P5 · P95", f"{p5 / div:,.1f} – {p95 / div:,.1f}", help="Caso adverso y caso favorable (US$ " + suf + ").")
    m[5].metric("Prob. de pérdida", f"{np.mean(x < 0):.2%}")

    st.markdown("##### 🎚️ Barra de probabilidad: elige un rango de utilidad y mira qué tan probable es")
    lo_min, hi_max = float(x.min()) / div, float(x.max()) / div
    paso = max((hi_max - lo_min) / 400, 1e-6)
    ini = (float(np.clip(umbral / div, lo_min, hi_max)), hi_max)
    rango = st.slider(f"Rango de utilidad (US$ {suf})", lo_min, hi_max, ini, step=paso,
                      key=f"rng_{n}_{int(umbral)}", format="%.2f")
    lo, hi = rango[0] * div, rango[1] * div
    dentro = float(np.mean((x >= lo) & (x <= hi)))
    c = st.columns(3)
    c[0].metric("Probabilidad dentro del rango", f"{dentro:.1%}")
    c[1].metric("Por debajo del rango", f"{np.mean(x < lo):.1%}")
    c[2].metric("Por encima del rango", f"{np.mean(x > hi):.1%}")

    conf = st.slider("O al revés: nivel de confianza deseado (%)", 50, 99, 80, key=f"conf_{n}",
                     help="Monto de utilidad que se supera con esa probabilidad.")
    st.info(md(f"Con **{conf}% de confianza**, la utilidad de **{n}** será de al menos "
               f"**US$ {np.percentile(x, 100 - conf) / div:,.2f} {suf}** "
               f"({np.percentile(x, 100 - conf) / det_u:.0%} del plan)." if det_u > 0 else
               f"Con **{conf}% de confianza**, la utilidad de **{n}** será de al menos "
               f"**US$ {np.percentile(x, 100 - conf) / div:,.2f} {suf}**."))

    a, b = st.columns(2)
    with a:
        st.plotly_chart(g.fig_histograma(x, (lo, hi), det_u, g.color_de(i), f"Distribución de la utilidad · {n}",
                                         umbral=umbral), width="stretch", key=f"hist_{n}")
    with b:
        sens = sensibilidad(df)
        if sens.empty:
            st.info("No hay variables inciertas para mostrar sensibilidad.")
        else:
            st.plotly_chart(g.fig_tornado(sens), width="stretch", key=f"torn_{n}")

    st.markdown("**Por zona: dónde se gana o se pierde respecto al plan**")
    tz = tabla_zonas(entrada, df)
    st.plotly_chart(g.fig_zonas(tz["Zona"].tolist(), tz["Utilidad plan (US$)"].tolist(),
                                tz["Utilidad media (US$)"].tolist(), g.color_de(i), div, suf), width="stretch",
                    key=f"zonas_{n}")
    mostrar = tz.copy()
    for c_ in ["Utilidad plan (US$)", "Utilidad media (US$)", "P5 (US$)", "P95 (US$)"]:
        mostrar[c_.replace("US$", f"US$ {suf}")] = mostrar.pop(c_) / div
    st.dataframe(mostrar, hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="%.2f") for k in mostrar.columns if k != "Zona"})
    st.caption(f"Onzas recuperables: plan {ctx['tot'][n][COL_O]:,.0f} oz · media simulada {df[COL_O].mean():,.0f} oz "
               f"(P5 {df[COL_O].quantile(.05):,.0f} · P95 {df[COL_O].quantile(.95):,.0f}).")
