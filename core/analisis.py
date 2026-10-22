"""Lectura automática de los resultados (texto) para la pantalla de comparación y el informe PDF."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .modelo import COL_CERTEZA, COL_LEY, ley_estimada


def _m(v: float) -> str:
    """Dinero legible: 24,10 M / 850 k."""
    a = abs(v)
    if a >= 1e6:
        return f"US$ {v / 1e6:,.2f} M"
    if a >= 1e3:
        return f"US$ {v / 1e3:,.1f} k"
    return f"US$ {v:,.0f}"


def criterio_texto(modo: str, valor: float) -> str:
    return f"alcanzar al menos el {valor:.0f} % de la utilidad planificada" if modo == "pct" \
        else f"alcanzar al menos {_m(valor)} de utilidad"


def veredicto(tabla: pd.DataFrame, modo: str, valor: float, certeza_pond: dict[str, float], n_iter: int) -> dict:
    """
    Devuelve {'recomendado': nombre, 'titulo': str, 'frases': [str,...], 'avisos': [str,...]}.
    `tabla` es la salida de estadistica.resumen(); `certeza_pond` = certeza ponderada del plan de cada escenario.
    """
    t = tabla.set_index("Escenario")
    prob = t["Prob. ≥ umbral"]
    mejor = str(prob.idxmax())
    orden = prob.sort_values(ascending=False)
    plan = t["Determinístico (US$)"]
    top_plan = str(plan.idxmax())
    cumpl = t["Cumplimiento esperado"]
    se = lambda p: float(np.sqrt(max(p * (1 - p), 1e-12) / n_iter))  # noqa: E731

    frases = [
        f"Con el criterio «{criterio_texto(modo, valor)}», **{mejor}** es el escenario con mayor probabilidad de éxito: "
        f"{prob[mejor]:.1%}."
    ]
    if len(orden) > 1:
        resto = ", ".join(f"{n} {p:.1%}" for n, p in orden.iloc[1:].items())
        frases.append(f"Le siguen: {resto}.")
    if top_plan != mejor:
        frases.append(
            f"**{top_plan}** tiene la mayor utilidad en el plan determinístico ({_m(plan[top_plan])}), pero solo "
            f"{prob[top_plan]:.1%} de probabilidad de cumplir el criterio, frente a {prob[mejor]:.1%} de {mejor} "
            f"(utilidad planificada {_m(plan[mejor])}). Su certeza geo-operativa ponderada en el plan es "
            f"{certeza_pond.get(top_plan, float('nan')):.0f} %, frente a {certeza_pond.get(mejor, float('nan')):.0f} % de {mejor}."
        )
    else:
        frases.append(f"{mejor} también es el de mayor utilidad en el plan: la incertidumbre de la certeza no cambia la elección.")
    if cumpl.notna().all():
        frases.append(
            "Cumplimiento esperado (utilidad media simulada ÷ plan): "
            + "; ".join(f"{n} {v:.0%}" for n, v in cumpl.items()) + "."
        )
    if "Certeza simulada media (%)" in t:
        frases.append(
            "Certeza geo-operativa ponderada simulada (media) frente a la del plan: "
            + "; ".join(f"{n} {t.loc[n, 'Certeza simulada media (%)']:.1f} % (plan {certeza_pond.get(n, float('nan')):.1f} %)"
                        for n in t.index) + "."
        )
    avisos = []
    if len(orden) > 1:
        p1, p2 = float(orden.iloc[0]), float(orden.iloc[1])
        if p1 - p2 < 2 * np.hypot(se(p1), se(p2)):
            avisos.append(
                "La diferencia entre los dos primeros está dentro del ruido de la simulación: "
                "aumenta las iteraciones o considera otros criterios antes de decidir."
            )
    if float(orden.iloc[0]) < 0.5:
        avisos.append("Ningún escenario supera el 50 % de probabilidad con este criterio: revisa si la meta es realista.")
    avisos.append(
        "El resultado depende del rango y de la distribución con que se sortea la certeza geo-operativa, que son supuestos "
        "de trabajo a validar con el criterio del ingeniero y datos de perforación."
    )
    return {"recomendado": mejor, "titulo": f"Escenario con mayor probabilidad de éxito: {mejor}",
            "frases": frases, "avisos": avisos}


def tabla_zonas(entrada: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Plan vs simulación por zona (utilidad, ley estimada, certeza)."""
    filas = []
    for _, z in entrada.iterrows():
        nombre = str(z["Zona"])
        ley_plan = float(ley_estimada(z[COL_LEY], z[COL_CERTEZA] / 100.0))
        oz = z["TMS"] * ley_plan * z["Recuperación (%)"] / 100 / 31.1035
        util_plan = oz * z["Precio (US$/oz)"] - z["Costo explotación (US$/t)"] * z["TMS"]
        ley_media = float(df[f"Ley estimada {nombre}"].mean())
        filas.append({
            "Zona": nombre, "Certeza plan (%)": float(z[COL_CERTEZA]),
            "Certeza simulada media (%)": float(df[f"Certeza {nombre}"].mean()),
            "Ley base (g/t)": float(z[COL_LEY]), "Ley estimada plan (g/t)": ley_plan,
            "Ley estimada media (g/t)": ley_media,
            "Variación de ley vs plan (%)": (ley_media / ley_plan - 1) * 100 if ley_plan else 0.0,
            "Utilidad plan (US$)": float(util_plan), "Utilidad media (US$)": float(df[f"Utilidad {nombre}"].mean()),
            "P5 (US$)": float(df[f"Utilidad {nombre}"].quantile(0.05)),
            "P95 (US$)": float(df[f"Utilidad {nombre}"].quantile(0.95)),
        })
    return pd.DataFrame(filas)
