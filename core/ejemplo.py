"""Datos de ejemplo: los 3 escenarios del Excel «Escenarios para Montecarlo» (mismas 3 zonas y misma certeza geo-operativa;
distinto tonelaje y ley base por zona y escenario). Precio, recuperación y costos son los del Excel."""
import pandas as pd

from .modelo import (
    COL_CERTEZA, COL_COSTO, COL_LEY, COL_PRECIO, COL_REC, COL_TMS, COL_ZONA, COLUMNAS_ENTRADA,
)

CERTEZA_ZONAS = [70.0, 88.0, 97.0]     # igual en los 3 escenarios (Excel, fila «Certeza Geo-Operativa»)
RECUPERACION = [95.0, 91.0, 93.0]
COSTOS = [80.0, 60.0, 40.0]


def tabla_base(tms: list[float], ley_base: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_ZONA: ["Zona 1", "Zona 2", "Zona 3"],
            COL_TMS: [float(t) for t in tms],
            COL_LEY: [float(x) for x in ley_base],
            COL_REC: RECUPERACION,
            COL_PRECIO: [4500.0, 4500.0, 4500.0],
            COL_COSTO: COSTOS,
            COL_CERTEZA: CERTEZA_ZONAS,
        }
    )[COLUMNAS_ENTRADA]


def escenarios_ejemplo() -> dict[str, pd.DataFrame]:
    return {
        "Escenario 1": tabla_base([5000, 10000, 15000], [7.0, 5.0, 6.0]),
        "Escenario 2": tabla_base([10000, 7000, 13000], [6.0, 5.0, 9.0]),
        "Escenario 3": tabla_base([15000, 3000, 12000], [5.0, 9.0, 10.0]),
    }


def tabla_vacia() -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_ZONA: ["Zona 1"],
            COL_TMS: [10000.0],
            COL_LEY: [5.0],
            COL_REC: [91.0],
            COL_PRECIO: [4500.0],
            COL_COSTO: [60.0],
            COL_CERTEZA: [88.0],
        }
    )[COLUMNAS_ENTRADA]
