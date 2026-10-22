"""Estadísticas de decisión: percentiles, probabilidades, tornado, comparación entre escenarios."""
from __future__ import annotations

import numpy as np
import pandas as pd

COL_U = "Utilidad (US$)"


def percentiles(x: np.ndarray, ps=(5, 10, 50, 90, 95)) -> dict[int, float]:
    return {p: float(np.percentile(x, p)) for p in ps}


def prob_mayor_igual(x: np.ndarray, umbral: float) -> float:
    return float(np.mean(np.asarray(x) >= umbral))


def prob_cumplimiento(x: np.ndarray, plan: float, pct: float) -> float:
    """Probabilidad de que la utilidad alcance al menos `pct` % de lo planificado (determinístico)."""
    return prob_mayor_igual(x, plan * pct / 100.0)


def curva_cumplimiento(x: np.ndarray, plan: float, pcts) -> np.ndarray:
    """P(utilidad ≥ X % del plan) para cada X de `pcts`."""
    x = np.sort(np.asarray(x))
    umbrales = plan * np.asarray(list(pcts), dtype=float) / 100.0
    return 1.0 - np.searchsorted(x, umbrales, side="left") / len(x)


def umbral_escenario(det_utilidad: float, modo: str, valor: float) -> float:
    """
    modo 'pct'   -> valor es el % del determinístico del escenario (100 = igualarlo)
    modo 'monto' -> valor es un monto fijo (US$) para todos los escenarios
    """
    return det_utilidad * valor / 100.0 if modo == "pct" else float(valor)


def resumen(resultados: dict[str, pd.DataFrame], det: dict[str, float], modo: str, valor: float) -> pd.DataFrame:
    """Tabla comparativa de todos los escenarios."""
    mejor = prob_mejor(resultados)
    filas = []
    for nom, df in resultados.items():
        x = df[COL_U].to_numpy()
        p = percentiles(x)
        um = umbral_escenario(det[nom], modo, valor)
        filas.append(
            {
                "Escenario": nom,
                "Determinístico (US$)": det[nom],
                "Media (US$)": float(x.mean()),
                "Cumplimiento esperado": float(x.mean() / det[nom]) if det[nom] > 0 else float("nan"),
                "Desv. estándar (US$)": float(x.std(ddof=1)),
                "P5": p[5],
                "P10": p[10],
                "P50": p[50],
                "P90": p[90],
                "P95": p[95],
                "Umbral (US$)": um,
                "Prob. ≥ umbral": prob_mayor_igual(x, um),
                "Prob. de pérdida (<0)": float(np.mean(x < 0)),
                "Prob. de ser el mejor": mejor[nom],
                "Onzas medias (oz)": float(df["Onzas (oz)"].mean()),
                "Certeza simulada media (%)": float(df["Certeza escenario (%)"].mean()),
                "Certeza simulada P5 (%)": float(df["Certeza escenario (%)"].quantile(0.05)),
                "Certeza simulada P95 (%)": float(df["Certeza escenario (%)"].quantile(0.95)),
                "Error estándar de la media (US$)": float(x.std(ddof=1) / np.sqrt(len(x))),
            }
        )
    return pd.DataFrame(filas)


def prob_mejor(resultados: dict[str, pd.DataFrame]) -> dict[str, float]:
    """En qué fracción de iteraciones cada escenario tiene la mayor utilidad."""
    nombres = list(resultados)
    m = np.column_stack([resultados[n][COL_U].to_numpy() for n in nombres])
    ganador = m.argmax(axis=1)
    return {n: float(np.mean(ganador == i)) for i, n in enumerate(nombres)}


def sensibilidad(df: pd.DataFrame, objetivo: str = COL_U) -> pd.DataFrame:
    """
    Correlación de rangos de Spearman entre la certeza de cada zona (la única variable incierta)
    y la utilidad: la base del gráfico tornado de @RISK.
    """
    entradas = [c for c in df.columns if c.startswith("Certeza ") and c != "Certeza escenario (%)"]
    rangos = df[entradas + [objetivo]].rank()
    filas = []
    for c in entradas:
        if rangos[c].std() > 0:
            filas.append({"Variable": c, "Correlación": float(rangos[c].corr(rangos[objetivo]))})
    out = pd.DataFrame(filas)
    if out.empty:
        return out
    return out.sort_values("Correlación", key=np.abs, ascending=True).reset_index(drop=True)
