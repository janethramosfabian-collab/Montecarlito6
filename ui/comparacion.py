"""Pestaña 4: todos los escenarios lado a lado, con la misma escala, para decidir."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.analisis import tabla_zonas, veredicto
from core.estadistica import COL_U
from . import graficos as g
from .resultados import aviso_desactualizado, contexto, md

POR_FILA = 3


def _filas(orden: list[str]):
    """Reparte los escenarios en filas de POR_FILA columnas iguales (simetría visual)."""
    for k in range(0, len(orden), POR_FILA):
        grupo = orden[k:k + POR_FILA]
        cols = st.columns(POR_FILA)
        for j, n in enumerate(grupo):
            yield cols[j], n, k + j


def render_comparacion() -> None:
    ctx = contexto()
    if ctx is None:
        st.info("Simula primero con **▶ Simular** en la barra lateral para comparar los escenarios.")
        return
    aviso_desactualizado(ctx["sim"])
    orden, res, det_u, div, suf = ctx["orden"], ctx["res"], ctx["det_u"], ctx["div"], ctx["suf"]
    tabla, modo, valor, umbrales = ctx["tabla"], ctx["modo"], ctx["valor"], ctx["umbrales"]
    t = tabla.set_index("Escenario")
    ver = veredicto(tabla, modo, valor, ctx["certeza_pond"], ctx["s"].n_iter)
    st.session_state["veredicto"] = ver

    # ---- veredicto
    with st.container(border=True):
        st.markdown(f"### 🏆 {ver['titulo']}")
        for f in ver["frases"]:
            st.markdown("- " + md(f))
        for a in ver["avisos"]:
            st.caption("⚠️ " + md(a))

    # ---- fichas simétricas
    st.markdown("#### Ficha de cada escenario")
    for col, n, i in _filas(orden):
        with col, st.container(border=True):
            c = g.color_de(i)
            corona = " 🏆" if n == ver["recomendado"] else ""
            st.markdown(f"<div style='border-left:6px solid {c};padding-left:10px;font-weight:700;font-size:1.1rem'>"
                        f"{n}{corona}</div>", unsafe_allow_html=True)
            r = t.loc[n]
            a, b = st.columns(2)
            a.metric("Prob. de éxito", f"{r['Prob. ≥ umbral']:.1%}")
            b.metric("Cumpl. esperado", f"{r['Cumplimiento esperado']:.0%}" if det_u[n] > 0 else "—")
            a.metric("Plan", f"{det_u[n] / div:,.2f} {suf}")
            b.metric("Media simulada", f"{r['Media (US$)'] / div:,.2f} {suf}")
            a.metric("P5 – P95", f"{r['P5'] / div:,.1f} – {r['P95'] / div:,.1f}")
            b.metric("Prob. de pérdida", f"{r['Prob. de pérdida (<0)']:.1%}")
            a.metric("Certeza pond. simulada", f"{r['Certeza simulada media (%)']:.1f} %", f"plan {ctx['certeza_pond'][n]:.1f} %",
                     delta_color="off", delta_arrow="off")
            b.metric("Onzas medias", f"{r['Onzas medias (oz)']:,.0f}", f"plan {ctx['tot'][n]['Onzas (oz)']:,.0f}", delta_color="off", delta_arrow="off")

    # ---- distribución con el mismo eje
    st.markdown("#### Distribución de la utilidad (mismo eje para todos)")
    todo = np.concatenate([res[n][COL_U].to_numpy() for n in orden])
    xr = (float(np.percentile(todo, 0.2)), float(np.percentile(todo, 99.8)))
    xr = (min(xr[0], min(det_u.values()) * 0.98), max(xr[1], max(det_u.values()) * 1.02))
    for col, n, i in _filas(orden):
        with col:
            x = res[n][COL_U].to_numpy()
            rango = (float(umbrales[n]), xr[1])
            st.plotly_chart(g.fig_histograma(x, rango, det_u[n], g.color_de(i), n, xrange=xr, umbral=umbrales[n], alto=300),
                            width="stretch", key=f"cmp_hist_{n}")
    st.caption("Barras de color intenso = iteraciones que cumplen el criterio (las tenues no). Línea amarilla = plan; línea roja punteada = umbral.")

    # ---- origen de la pérdida por zona (misma escala)
    st.markdown("#### ¿Dónde se gana o se pierde valor? Plan vs. media simulada por zona")
    tzs = {n: tabla_zonas(ctx["entradas"][n], res[n]) for n in orden}
    ymax = max(max(tz["Utilidad plan (US$)"].max(), tz["Utilidad media (US$)"].max()) for tz in tzs.values())
    ymin = min(0.0, min(min(tz["Utilidad plan (US$)"].min(), tz["Utilidad media (US$)"].min()) for tz in tzs.values()))
    for col, n, i in _filas(orden):
        with col:
            tz = tzs[n]
            st.plotly_chart(g.fig_zonas(tz["Zona"].tolist(), tz["Utilidad plan (US$)"].tolist(),
                                        tz["Utilidad media (US$)"].tolist(), g.color_de(i), div, suf,
                                        yrange=(ymin * 1.1, ymax * 1.2), titulo=n),
                            width="stretch", key=f"cmp_zonas_{n}")

    # ---- certeza geo-operativa del escenario: plan vs simulada
    st.markdown("#### Certeza geo-operativa de cada escenario: plan vs. simulada")
    st.plotly_chart(
        g.fig_certeza(orden, [ctx["certeza_pond"][n] for n in orden], t["Certeza simulada media (%)"].tolist(),
                      t["Certeza simulada P5 (%)"].tolist(), t["Certeza simulada P95 (%)"].tolist()),
        width="stretch", key="cmp_certeza")
    st.caption("Rombo = certeza ponderada por TMS en el plan; punto = media de la certeza simulada; barra = rango P5–P95. "
               "Si el plan queda por encima de la media simulada, el escenario necesita más certeza de la que se espera tener.")

    # ---- gráficos globales
    st.markdown("#### Todos juntos")
    a, b = st.columns(2)
    with a:
        pct_actual = valor if modo == "pct" else None
        st.plotly_chart(g.fig_curva_cumplimiento(res, det_u, orden, pct_actual), width="stretch", key="cmp_curva")
        st.caption("Cuanto más arriba y a la derecha esté la curva de un escenario, más probable es que cumpla "
                   "una meta exigente. Donde las curvas se cruzan, cambia cuál es mejor.")
    with b:
        st.plotly_chart(
            g.fig_probabilidad(orden, [det_u[n] for n in orden], t["P5"].tolist(), t["P95"].tolist(),
                               t["Prob. ≥ umbral"].tolist(), [umbrales[n] for n in orden]),
            width="stretch", key="cmp_prob")
    st.plotly_chart(g.fig_scurves(res, COL_U, umbrales, orden), width="stretch", key="cmp_scurve")

    # ---- tabla numérica
    st.markdown("#### Tabla comparativa")
    cols_dinero = ["Determinístico (US$)", "Media (US$)", "P5", "P50", "P95"]
    tab = pd.DataFrame({
        "Escenario": tabla["Escenario"],
        f"Plan (US$ {suf})": tabla["Determinístico (US$)"] / div,
        f"Media (US$ {suf})": tabla["Media (US$)"] / div,
        "Cumplimiento esperado (%)": tabla["Cumplimiento esperado"] * 100,
        "Prob. de éxito (%)": tabla["Prob. ≥ umbral"] * 100,
        f"P5 (US$ {suf})": tabla["P5"] / div,
        f"P50 (US$ {suf})": tabla["P50"] / div,
        f"P95 (US$ {suf})": tabla["P95"] / div,
        "Prob. de pérdida (%)": tabla["Prob. de pérdida (<0)"] * 100,
        "Prob. de ser el mejor (%)": tabla["Prob. de ser el mejor"] * 100,
        "Onzas plan (oz)": [ctx["tot"][n]["Onzas (oz)"] for n in tabla["Escenario"]],
        "Onzas medias (oz)": tabla["Onzas medias (oz)"],
        "Certeza plan (%)": [ctx["certeza_pond"][n] for n in tabla["Escenario"]],
        "Certeza simulada media (%)": tabla["Certeza simulada media (%)"],
    })
    mejores = ["Prob. de éxito (%)", "Cumplimiento esperado (%)", f"Media (US$ {suf})", f"P5 (US$ {suf})"]
    fmt = {c: "{:,.2f}" for c in tab.columns[1:]}
    for c in ["Onzas plan (oz)", "Onzas medias (oz)"]:
        fmt[c] = "{:,.0f}"
    for c in ["Cumplimiento esperado (%)", "Prob. de éxito (%)", "Prob. de pérdida (%)", "Prob. de ser el mejor (%)",
              "Certeza plan (%)", "Certeza simulada media (%)"]:
        fmt[c] = "{:,.1f}"
    sty = tab.style.format(fmt).highlight_max(subset=mejores, color="#1f5f46", axis=0)
    st.dataframe(sty, hide_index=True, width="stretch")
    st.caption(f"Verde = mejor valor de la columna. {ctx['s'].n_iter:,} iteraciones · semilla {ctx['s'].semilla} · error estándar "
               f"de la probabilidad ≤ {50 / np.sqrt(ctx['s'].n_iter):.1f} puntos. Diferencias menores que eso son ruido de simulación.")
