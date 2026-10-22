"""Exportación a Excel (supuestos, determinístico, estadísticas y simulaciones)."""
from __future__ import annotations

import io

import pandas as pd

from .modelo import Supuestos, tabla_determinista, totales_deterministas


def tabla_supuestos(s: Supuestos) -> pd.DataFrame:
    """Supuestos en formato legible para replicar el modelo en @RISK."""
    filas = [
        ("Iteraciones", s.n_iter),
        ("Semilla", s.semilla),
        ("Variable incierta", "Certeza geo-operativa de cada zona (única variable que se sortea)"),
        ("Precio, TMS, ley base, recuperación y costos", "Fijos (valores de la tabla)"),
        ("Certeza - distribución", s.certeza_dist),
        ("Certeza - mínimo (%)", s.certeza_min_pct),
        ("Certeza - máximo (%)", s.certeza_max_pct),
    ]
    if s.certeza_dist in ("Triangular", "PERT"):
        filas.append(("Certeza - más probable", "La certeza de la tabla de cada zona (acotada al rango)"))
    if s.certeza_dist == "Normal":
        filas.append(("Certeza - desv. estándar (puntos %)", s.certeza_sd_pp))
    filas += [
        ("Regla certeza → ley", "ley estimada = ley base × (2 − 1/certeza)"),
        ("Correlación entre zonas (ρ)", s.rho_zonas),
        ("Misma zona = mismo sorteo entre escenarios", "Sí" if s.compartir_geologia else "No"),
    ]
    return pd.DataFrame(filas, columns=["Supuesto", "Valor"]).astype({"Valor": "string"})


def excel_bytes(
    entradas: dict[str, pd.DataFrame],
    supuestos: Supuestos,
    resumen: pd.DataFrame | None,
    resultados: dict[str, pd.DataFrame] | None,
    max_filas_sim: int = 20000,
) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        tabla_supuestos(supuestos).to_excel(xw, sheet_name="Supuestos", index=False)
        bloques = []
        for nom, df in entradas.items():
            det = tabla_determinista(df)
            tot = totales_deterministas(df)
            fila_tot = {"Zona": "TOTAL", **{k: v for k, v in tot.items()}}
            det = pd.concat([det, pd.DataFrame([fila_tot])], ignore_index=True)
            det.insert(0, "Escenario", nom)
            bloques.append(det)
        pd.concat(bloques, ignore_index=True).to_excel(xw, sheet_name="Determinístico", index=False)
        if resumen is not None:
            resumen.to_excel(xw, sheet_name="Resumen estocástico", index=False)
        if resultados:
            for nom, df in resultados.items():
                df.head(max_filas_sim).to_excel(xw, sheet_name=f"Sim {nom}"[:31], index=False)
    return buf.getvalue()
