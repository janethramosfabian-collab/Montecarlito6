"""Pestaña 6: informe PDF con firma, Excel y CSV."""
from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from core.analisis import veredicto
from core.exportar import excel_bytes
from core.informe_pdf import pdf_bytes
from .resultados import aviso_desactualizado, contexto


def render_exportar(entradas: dict[str, pd.DataFrame]) -> None:
    ctx = contexto()
    if ctx is None:
        st.info("Primero corre una simulación con **▶ Simular** en la barra lateral.")
        return
    aviso_desactualizado(ctx["sim"])
    res, s, sim = ctx["res"], ctx["s"], ctx["sim"]

    st.subheader("📄 Informe técnico en PDF")
    st.caption("Incluye resumen ejecutivo, método, datos, gráficos, análisis técnico automático, limitaciones y bloque de firmas. "
               "Usa el criterio de éxito elegido en la barra lateral.")
    with st.container(border=True):
        a, b = st.columns(2)
        proyecto = a.text_input("Proyecto", value="Proyecto minero", key="pdf_proy")
        empresa = b.text_input("Empresa / área (opcional)", key="pdf_emp")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Elaborado por**")
            el_nom = st.text_input("Nombre", key="pdf_el_nom")
            el_car = st.text_input("Cargo", key="pdf_el_car")
            f1 = st.file_uploader("Imagen de la firma (opcional, PNG/JPG)", type=["png", "jpg", "jpeg"], key="pdf_f1")
        with c2:
            st.markdown("**Revisado y aprobado por**")
            rv_nom = st.text_input("Nombre ", key="pdf_rv_nom")
            rv_car = st.text_input("Cargo ", key="pdf_rv_car")
            f2 = st.file_uploader("Imagen de la firma (opcional, PNG/JPG) ", type=["png", "jpg", "jpeg"], key="pdf_f2")
    meta = {"proyecto": proyecto.strip(), "empresa": empresa.strip(), "elaborado_por": el_nom.strip(), "cargo": el_car.strip(),
            "revisado_por": rv_nom.strip(), "cargo_rev": rv_car.strip(),
            "firma1": f1.getvalue() if f1 else None, "firma2": f2.getvalue() if f2 else None}
    clave = hashlib.md5((sim["firma"] + repr(ctx["modo"]) + repr(ctx["valor"]) +
                         repr({k: (v if not isinstance(v, bytes) else hashlib.md5(v).hexdigest()) for k, v in meta.items()})).encode()).hexdigest()
    if st.button("Generar informe PDF", type="primary", key="gen_pdf"):
        with st.spinner("Generando PDF (gráficos y análisis)…"):
            ver = veredicto(ctx["tabla"], ctx["modo"], ctx["valor"], ctx["certeza_pond"], s.n_iter)
            data = pdf_bytes(entradas=ctx["entradas"], supuestos=s, res=res, tot=ctx["tot"], det_u=ctx["det_u"],
                             tabla=ctx["tabla"], modo=ctx["modo"], valor=ctx["valor"], umbrales=ctx["umbrales"], ver=ver,
                             huella=sim["firma"], meta=meta)
            st.session_state["pdf"] = (clave, data)
    paquete = st.session_state.get("pdf")
    if paquete and paquete[0] == clave:
        st.download_button("⬇️ Descargar informe PDF", paquete[1], file_name="informe_montecarlo.pdf", mime="application/pdf",
                           key="dl_pdf")
    elif paquete:
        st.caption("Cambiaron los datos, el criterio o los campos: vuelve a generar el PDF.")

    st.divider()
    st.subheader("📤 Datos")
    st.caption("El Excel incluye supuestos, determinístico, resumen estocástico y las primeras 20 000 iteraciones de cada escenario.")
    if st.button("Preparar archivo Excel", key="prep_xlsx"):
        with st.spinner("Generando Excel…"):
            st.session_state["xlsx"] = (sim["firma"], excel_bytes(ctx["entradas"], s, ctx["tabla"], res))
    paquete = st.session_state.get("xlsx")
    if paquete and paquete[0] == sim["firma"]:
        st.download_button("⬇️ Descargar Excel", paquete[1], file_name="simulacion_montecarlo.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_xlsx")
    nombre = st.selectbox("CSV de iteraciones de un escenario", list(res), key="csv_esc")
    st.download_button("⬇️ Descargar CSV", res[nombre].to_csv(index=False).encode("utf-8"),
                       file_name=f"iteraciones_{nombre}.csv", mime="text/csv", key="dl_csv")
