"""
Modelo minero por zonas y simulación de Montecarlo (versión 3).

Determinístico (idéntico a tu Excel, por zona) = el PLAN:
    Ley estimada = Ley base + (Ley base - Ley base / Certeza)  =  Ley base * (2 - 1/Certeza)
    Onzas        = TMS * Ley estimada * Recuperación / 31.1035
    Valorizado   = Precio * Onzas
    Costo total  = Costo explotación * TMS
    Utilidad     = Valorizado - Costo total

Totales (ponderados por TMS de ESE escenario):
    TMS, Onzas, Valorizado, Costo, Utilidad = sumas (Utilidad = Valorizado - Costo)
    Ley base, Ley estimada, Recuperación, Precio, Certeza = SUMPRODUCT(TMS, x) / TMS total

Estocástico: la ÚNICA variable que se sortea en cada iteración es la CERTEZA GEO-OPERATIVA de cada zona,
dentro de un rango común (70 % a 97 % por defecto). El precio, las TMS, la ley base, la recuperación y los
costos se mantienen fijos. La ley estimada ya incluye la certeza, así que al variar la certeza varían la ley,
las onzas y la utilidad. El plan determinístico usa la certeza de la tabla; se mide qué tan probable es cumplirlo.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

G_POR_OZ = 31.1035

COL_ZONA = "Zona"
COL_TMS = "TMS"
COL_LEY = "Ley base (g/t)"
COL_LEY_EST = "Ley estimada (g/t)"
COL_REC = "Recuperación (%)"
COL_PRECIO = "Precio (US$/oz)"
COL_COSTO = "Costo explotación (US$/t)"
COL_CERTEZA = "Certeza geo-operativa (%)"
COLUMNAS_ENTRADA = [COL_ZONA, COL_TMS, COL_LEY, COL_REC, COL_PRECIO, COL_COSTO, COL_CERTEZA]

DIST_CERTEZA = ["Triangular", "PERT", "Uniforme", "Normal"]
CERTEZA_MIN_DEFECTO = 70.0   # % : límite inferior del sorteo de la certeza geo-operativa
CERTEZA_MAX_DEFECTO = 97.0   # % : límite superior
ITERACIONES = [20000, 30000, 50000, 100000]


# --------------------------------------------------------------------------
# Supuestos
# --------------------------------------------------------------------------
@dataclass
class Supuestos:
    n_iter: int = 20000
    semilla: int = 42
    # Certeza geo-operativa: se sortea en [min, max] (%), igual para todas las zonas y escenarios.
    # Triangular / PERT: el valor más probable de cada zona es su certeza de la tabla (acotada al rango).
    certeza_dist: str = "Triangular"
    certeza_min_pct: float = CERTEZA_MIN_DEFECTO
    certeza_max_pct: float = CERTEZA_MAX_DEFECTO
    certeza_sd_pp: float = 7.0       # solo Normal: desviación estándar (puntos %) alrededor de la certeza de la tabla
    # Correlación entre la certeza de distintas zonas (0 = independientes)
    rho_zonas: float = 0.0
    # Misma zona = mismo sorteo en todos los escenarios (comparación justa)
    compartir_geologia: bool = True

    def a_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Regla certeza → ley (la del Excel)
# --------------------------------------------------------------------------
def factor_ley(certeza):
    """Fracción de la ley base que se conserva: 2 - 1/certeza (certeza en 0-1). Nunca negativa.
    Equivale a la fórmula del Excel: ley estimada = ley base + (ley base - ley base / certeza)."""
    c = np.maximum(np.asarray(certeza, dtype=float), 1e-9)
    return np.clip(2.0 - 1.0 / c, 0.0, None)


def ley_estimada(ley_base, certeza):
    return np.asarray(ley_base, dtype=float) * factor_ley(certeza)


# --------------------------------------------------------------------------
# Validación y determinístico
# --------------------------------------------------------------------------
def limpiar_tabla(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza la tabla editada: tipos numéricos y sin filas vacías."""
    d = df.copy()
    for c in COLUMNAS_ENTRADA:
        if c not in d.columns:
            d[c] = np.nan
    d = d[COLUMNAS_ENTRADA]
    d[COL_ZONA] = d[COL_ZONA].astype("string").str.strip()
    for c in COLUMNAS_ENTRADA[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(how="all")
    d = d[d[COL_ZONA].notna() & (d[COL_ZONA] != "")]
    return d.reset_index(drop=True)


def validar_tabla(df: pd.DataFrame, nombre: str) -> list[str]:
    errores: list[str] = []
    if df.empty:
        return [f"{nombre}: agrega al menos una zona."]
    if df[COL_ZONA].duplicated().any():
        errores.append(f"{nombre}: hay zonas con el mismo nombre.")
    if df[COLUMNAS_ENTRADA[1:]].isna().any().any():
        errores.append(f"{nombre}: hay celdas vacías o no numéricas.")
    if (df[COL_TMS] < 0).any() or (df[COL_LEY] < 0).any() or (df[COL_COSTO] < 0).any() or (df[COL_PRECIO] < 0).any():
        errores.append(f"{nombre}: TMS, ley, costo y precio no pueden ser negativos.")
    if ((df[COL_REC] < 0) | (df[COL_REC] > 100)).any():
        errores.append(f"{nombre}: la recuperación debe estar entre 0 y 100 %.")
    if ((df[COL_CERTEZA] <= 0) | (df[COL_CERTEZA] > 100)).any():
        errores.append(f"{nombre}: la certeza geo-operativa debe ser mayor que 0 y como máximo 100 %.")
    if not errores and df[COL_TMS].sum() <= 0:
        errores.append(f"{nombre}: el tonelaje total debe ser mayor que cero.")
    return errores


def tabla_determinista(df: pd.DataFrame) -> pd.DataFrame:
    """Cálculo por zona, igual que las columnas del Excel (aquí las zonas van en filas)."""
    d = limpiar_tabla(df)
    tms, ley_base = d[COL_TMS], d[COL_LEY]
    cert = d[COL_CERTEZA] / 100.0
    ley_est = ley_estimada(ley_base, cert)
    rec, precio = d[COL_REC] / 100.0, d[COL_PRECIO]
    onzas = tms * ley_est * rec / G_POR_OZ
    valorizado = onzas * precio
    costo_total = d[COL_COSTO] * tms
    return pd.DataFrame(
        {
            "Zona": d[COL_ZONA],
            "TMS": tms,
            COL_LEY: ley_base,
            COL_LEY_EST: ley_est,
            "Recuperación (%)": d[COL_REC],
            "Onzas (oz)": onzas,
            "Precio (US$/oz)": precio,
            "Valorizado (US$)": valorizado,
            "Costo explotación (US$/t)": d[COL_COSTO],
            "Costo total (US$)": costo_total,
            "Utilidad (US$)": valorizado - costo_total,
            COL_CERTEZA: d[COL_CERTEZA],
        }
    )


def totales_deterministas(df: pd.DataFrame) -> dict:
    """Columna 'Total' del Excel, ponderada por las TMS del propio escenario."""
    t = tabla_determinista(df)
    tms_total = float(t["TMS"].sum())

    def pond(col: str) -> float:
        return float((t["TMS"] * t[col]).sum() / tms_total) if tms_total else 0.0

    valorizado = float(t["Valorizado (US$)"].sum())
    costo = float(t["Costo total (US$)"].sum())
    return {
        "TMS": tms_total,
        COL_LEY: pond(COL_LEY),
        COL_LEY_EST: pond(COL_LEY_EST),
        "Recuperación (%)": pond("Recuperación (%)"),
        "Onzas (oz)": float(t["Onzas (oz)"].sum()),
        "Precio (US$/oz)": pond("Precio (US$/oz)"),
        "Valorizado (US$)": valorizado,
        "Costo total (US$)": costo,
        "Utilidad (US$)": valorizado - costo,
        COL_CERTEZA: pond(COL_CERTEZA),
    }


# --------------------------------------------------------------------------
# Muestreo
# --------------------------------------------------------------------------
def _uniformes_lhs(n: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube 1-D: una muestra por cada estrato de probabilidad."""
    u = (rng.permutation(n) + rng.random(n)) / n
    return np.clip(u, 1e-9, 1 - 1e-9)


def _ppf_triangular(u, lo, moda, hi):
    if hi <= lo:
        return np.full_like(u, lo, dtype=float)
    moda = min(max(moda, lo), hi)
    return stats.triang.ppf(u, c=(moda - lo) / (hi - lo), loc=lo, scale=hi - lo)


def _ppf_pert(u, lo, moda, hi):
    """Beta-PERT: como la triangular pero más concentrada en el valor más probable."""
    if hi <= lo:
        return np.full_like(u, lo, dtype=float)
    moda = min(max(moda, lo), hi)
    a = 1 + 4 * (moda - lo) / (hi - lo)
    b = 1 + 4 * (hi - moda) / (hi - lo)
    return stats.beta.ppf(u, a, b, loc=lo, scale=hi - lo)


def certeza_simulada(u: np.ndarray, base: float, s: Supuestos) -> np.ndarray:
    """
    Certeza geo-operativa (0-1) en cada iteración, dentro de [min, max] del supuesto.
    `base` = certeza de la tabla de la zona (0-1): es el valor más probable (Triangular/PERT) o el centro (Normal).
    """
    lo = max(s.certeza_min_pct / 100.0, 1e-6)
    hi = min(s.certeza_max_pct / 100.0, 1.0)
    if hi <= lo:
        return np.full_like(u, lo, dtype=float)
    moda = min(max(base, lo), hi)
    if s.certeza_dist == "Triangular":
        c = _ppf_triangular(u, lo, moda, hi)
    elif s.certeza_dist == "PERT":
        c = _ppf_pert(u, lo, moda, hi)
    elif s.certeza_dist == "Uniforme":
        c = lo + (hi - lo) * u
    elif s.certeza_dist == "Normal":
        sd = max(s.certeza_sd_pp / 100.0, 1e-6)
        c = stats.truncnorm.ppf(u, (lo - moda) / sd, (hi - moda) / sd, loc=moda, scale=sd)
    else:
        raise ValueError(f"Distribución de certeza desconocida: {s.certeza_dist}")
    return np.clip(np.asarray(c, dtype=float), lo, hi)


# --------------------------------------------------------------------------
# Simulación
# --------------------------------------------------------------------------
def simular(escenarios: dict[str, pd.DataFrame], s: Supuestos) -> dict[str, pd.DataFrame]:
    """
    Corre Montecarlo para todos los escenarios con los MISMOS números aleatorios
    (si `compartir_geologia`, misma zona = mismo sorteo de certeza en todos los escenarios).
    Solo la certeza geo-operativa es aleatoria; el precio y el resto de datos son los de la tabla.
    Devuelve un DataFrame por escenario con una fila por iteración.
    """
    n = int(s.n_iter)
    rng = np.random.default_rng(int(s.semilla))
    tablas = {nom: limpiar_tabla(df) for nom, df in escenarios.items()}

    z_global = rng.standard_normal(n)
    rho = float(np.clip(s.rho_zonas, 0.0, 0.99))

    sorteos: dict[tuple, np.ndarray] = {}

    def sorteo(clave: tuple) -> np.ndarray:
        """Uniformes (0-1) de la certeza de una zona; con ρ > 0 las zonas se correlacionan (cópula gaussiana)."""
        if clave not in sorteos:
            if rho > 0:
                z = np.sqrt(rho) * z_global + np.sqrt(1.0 - rho) * rng.standard_normal(n)
                sorteos[clave] = np.clip(stats.norm.cdf(z), 1e-9, 1 - 1e-9)
            else:
                sorteos[clave] = _uniformes_lhs(n, rng)
        return sorteos[clave]

    resultados: dict[str, pd.DataFrame] = {}
    for nom, t in tablas.items():
        util = np.zeros(n)
        onzas = np.zeros(n)
        cols: dict[str, np.ndarray] = {}
        for _, fila in t.iterrows():
            zona = str(fila[COL_ZONA])
            clave = (zona,) if s.compartir_geologia else (nom, zona)
            cert = certeza_simulada(sorteo(clave), float(fila[COL_CERTEZA]) / 100.0, s)
            ley_est = ley_estimada(fila[COL_LEY], cert)
            oz = fila[COL_TMS] * ley_est * (fila[COL_REC] / 100.0) / G_POR_OZ
            util_z = oz * fila[COL_PRECIO] - fila[COL_COSTO] * fila[COL_TMS]
            onzas += oz
            util += util_z
            cols[f"Certeza {zona}"] = cert * 100.0
            cols[f"Ley estimada {zona}"] = ley_est
            cols[f"Utilidad {zona}"] = util_z
        df = pd.DataFrame({"Utilidad (US$)": util, "Onzas (oz)": onzas, **cols})
        # Certeza geo-operativa del escenario (ponderada por TMS), como la fila del Excel
        pesos = t[COL_TMS].to_numpy() / t[COL_TMS].sum()
        df["Certeza escenario (%)"] = sum(
            p * df[f"Certeza {z}"] for p, z in zip(pesos, t[COL_ZONA].astype(str))
        )
        resultados[nom] = df
    return resultados
