"""Plataforma de simulación de Montecarlo para planeamiento minero."""
import hashlib
import json

import streamlit as st

from core import simular
from ui.datos import init_estado, render_escenarios, render_supuestos, cargar_ejemplo
from ui.comparacion import render_comparacion
from ui.exportar import render_exportar
from ui.resultados import render_criterio, render_resultados
from ui.teoria import render_teoria

st.set_page_config(page_title="Montecarlo Minero", page_icon="⛏️", layout="wide", initial_sidebar_state="expanded")
try:
    with open("assets/estilos.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except OSError:
    pass

init_estado()


def firma_de(entradas, supuestos) -> str:
    payload = {"e": {n: df.to_json() for n, df in entradas.items()}, "s": supuestos.a_dict()}
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()


st.title("⛏️ Montecarlo para planeamiento minero")
st.caption("Compara escenarios de producción con incertidumbre en la certeza geo-operativa y decide cuál tiene más probabilidad de cumplir el plan.")

t1, t2, t3, t4, t5, t6 = st.tabs(["1 · Escenarios", "2 · Incertidumbre", "3 · Resultados", "4 · Comparación",
                                  "5 · Teoría", "6 · Informe y exportar"])
with t1:
    entradas, errores = render_escenarios()
with t2:
    supuestos = render_supuestos(entradas)
errores = errores + st.session_state.get("errores_supuestos", [])
st.session_state["firma_actual"] = firma_de(entradas, supuestos) if not errores else None

with st.sidebar:
    st.header("Simulación")
    st.write(f"**{len(entradas)}** escenario(s) · **{supuestos.n_iter:,}** iteraciones")
    if errores:
        st.error("Corrige los errores marcados en las pestañas 1 y 2 para poder simular.")
    if st.button("▶ Simular", type="primary", width="stretch", disabled=bool(errores)):
        with st.spinner("Simulando…"):
            st.session_state["sim"] = {
                "firma": st.session_state["firma_actual"],
                "resultados": simular(entradas, supuestos),
                "entradas": {n: df.copy() for n, df in entradas.items()},
                "supuestos": supuestos,
            }
    sim = st.session_state.get("sim")
    if sim:
        vigente = sim["firma"] == st.session_state.get("firma_actual")
        if vigente:
            st.success("Resultados al día")
        else:
            st.warning("Hay cambios sin simular")
    st.divider()
    render_criterio(st.session_state.get("sim"))
    st.divider()
    st.caption("Los datos de ejemplo son los 3 escenarios de tu Excel. Todo es editable.")

with t3:
    if entradas and not errores:
        render_resultados(entradas)
    else:
        st.info("Completa los datos de los escenarios para ver resultados.")
with t4:
    if entradas and not errores:
        render_comparacion()
    else:
        st.info("Completa los datos de los escenarios para comparar.")
with t5:
    render_teoria()
with t6:
    render_exportar(entradas)
