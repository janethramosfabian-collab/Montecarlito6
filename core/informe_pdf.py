"""Informe técnico en PDF: resumen, método, resultados, análisis automático y bloque de firmas."""
from __future__ import annotations

import io
import math
import os
import re
from datetime import date
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from .analisis import tabla_zonas
from .estadistica import COL_U, curva_cumplimiento, sensibilidad
from .modelo import COL_CERTEZA, COL_LEY, COL_LEY_EST, Supuestos, factor_ley, tabla_determinista

# ---- tipografía (DejaVu viene con matplotlib: soporta ≥, →, σ, −)
_FD = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
for _n, _f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
               ("DV-BI", "DejaVuSans-BoldOblique.ttf")]:
    pdfmetrics.registerFont(TTFont(_n, os.path.join(_FD, _f)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

PAL = ["#2a6fd1", "#c4491a", "#14825c", "#a86f00", "#b83b6b", "#006b00", "#6a5fcf", "#cc4444"]
AZUL = colors.HexColor("#1f3a5f")
GRIS = colors.HexColor("#555555")
CLARO = colors.HexColor("#eef2f7")
VERDE = colors.HexColor("#d9efe5")

W = A4[0] - 4 * cm  # ancho útil

def _kn(est: ParagraphStyle) -> ParagraphStyle:
    est.keepWithNext = 1
    return est


ST = {
    "t": ParagraphStyle("t", fontName="DV-B", fontSize=20, leading=25, textColor=AZUL, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=10.5, leading=14, textColor=GRIS),
    "h1": _kn(ParagraphStyle("h1", fontName="DV-B", fontSize=13.5, leading=17, textColor=AZUL, spaceBefore=14, spaceAfter=6)),
    "h2": _kn(ParagraphStyle("h2", fontName="DV-B", fontSize=10.5, leading=14, textColor=AZUL, spaceBefore=8, spaceAfter=3)),
    "pk": _kn(ParagraphStyle("pk", fontName="DV", fontSize=9, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=4)),
    "p": ParagraphStyle("p", fontName="DV", fontSize=9, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=4),
    "b": ParagraphStyle("b", fontName="DV", fontSize=9, leading=13, leftIndent=12, bulletIndent=2, spaceAfter=2),
    "c": ParagraphStyle("c", fontName="DV-I", fontSize=7.8, leading=10.5, textColor=GRIS, spaceAfter=6),
    "tc": ParagraphStyle("tc", fontName="DV", fontSize=7.6, leading=9.6),
    "tch": ParagraphStyle("tch", fontName="DV-B", fontSize=7.6, leading=9.6, textColor=colors.white),
}


def _t(texto: str) -> str:
    """Texto con **negritas** → marcado de reportlab (escapando lo demás)."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(str(texto)))


def P(texto: str, estilo: str = "p") -> Paragraph:
    return Paragraph(_t(texto), ST[estilo])


def B(texto: str) -> Paragraph:
    return Paragraph(_t(texto), ST["b"], bulletText="•")


def _m(v: float) -> str:
    a = abs(v)
    if a >= 1e6:
        return f"US$ {v / 1e6:,.2f} M"
    if a >= 1e3:
        return f"US$ {v / 1e3:,.1f} k"
    return f"US$ {v:,.0f}"


# ------------------------------------------------------------------------------ gráficos
def _estilo(ax) -> None:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#999999")
    ax.grid(axis="y", color="#e3e3e3", lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors="#444444")


def _png(fig, ancho_cm: float = 17.0) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image as PI
    w, h = PI.open(io.BytesIO(buf.getvalue())).size
    buf.seek(0)
    ancho = ancho_cm * cm
    return Image(buf, width=ancho, height=ancho * h / w)


def _g_determinista(orden, det_u, tot):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
    div, suf = (1e6, "M") if max(det_u.values()) >= 1e6 else (1e3, "k")
    for i, n in enumerate(orden):
        a.bar(n, det_u[n] / div, color=PAL[i], width=0.5)
        a.text(i, det_u[n] / div, f"{det_u[n] / div:,.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        b.bar(n, tot[n]["Onzas (oz)"], color=PAL[i], width=0.5)
        b.text(i, tot[n]["Onzas (oz)"], f"{tot[n]['Onzas (oz)']:,.0f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    a.set_title(f"Utilidad planificada (US$ {suf})", fontsize=9.5, loc="left")
    b.set_title("Onzas recuperables planificadas (oz)", fontsize=9.5, loc="left")
    a.set_ylim(0, max(det_u.values()) / div * 1.18)
    b.set_ylim(0, max(tot[n]["Onzas (oz)"] for n in orden) * 1.18)
    for ax in (a, b):
        _estilo(ax)
    fig.tight_layout()
    return _png(fig)


def _g_exito(orden, tabla):
    t = tabla.set_index("Escenario")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
    for i, n in enumerate(orden):
        p = t.loc[n, "Prob. ≥ umbral"] * 100
        c = t.loc[n, "Cumplimiento esperado"] * 100
        a.bar(n, p, color=PAL[i], width=0.5)
        a.text(i, p, f"{p:.1f} %", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        b.bar(n, c, color=PAL[i], width=0.5)
        b.text(i, c, f"{c:.0f} %", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    a.set_ylim(0, 112)
    b.set_ylim(0, max(112, np.nanmax(t["Cumplimiento esperado"] * 100) * 1.12))
    b.axhline(100, color="#a86f00", lw=1.2, ls="--")
    a.set_title("Probabilidad de éxito (%)", fontsize=9.5, loc="left")
    b.set_title("Cumplimiento esperado: media ÷ plan (%)", fontsize=9.5, loc="left")
    for ax in (a, b):
        _estilo(ax)
    fig.tight_layout()
    return _png(fig)


def _g_curva(orden, res, det_u, modo, valor):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    xs = np.arange(0, 121, 1.0)
    for i, n in enumerate(orden):
        p = curva_cumplimiento(res[n][COL_U].to_numpy(), det_u[n], xs) * 100
        ax.plot(xs, p, color=PAL[i], lw=2.4, label=n)
        if modo == "pct":
            py = float(curva_cumplimiento(res[n][COL_U].to_numpy(), det_u[n], [valor])[0] * 100)
            ax.plot([valor], [py], "o", color=PAL[i], ms=6, mec="white")
            ax.annotate(f"{py:.0f} %", (valor, py), textcoords="offset points", xytext=(6, 2), fontsize=8, color=PAL[i])
    if modo == "pct":
        ax.axvline(valor, color="#444444", ls="--", lw=1)
    ax.axvline(100, color="#a86f00", lw=1.6)
    ax.text(100.8, 100, "Plan", color="#a86f00", fontsize=8, va="top")
    ax.set_xlabel("Utilidad alcanzada (% del plan)", fontsize=8.5)
    ax.set_ylabel("Probabilidad de alcanzarla o superarla (%)", fontsize=8.5)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 103)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    _estilo(ax)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    fig.tight_layout()
    return _png(fig)


def _g_histogramas(orden, res, det_u, umbrales):
    n = len(orden)
    cols = min(n, 3)
    filas = math.ceil(n / cols)
    todo = np.concatenate([res[k][COL_U].to_numpy() for k in orden])
    xr = (min(np.percentile(todo, 0.2), min(det_u.values()) * 0.98), max(np.percentile(todo, 99.8), max(det_u.values()) * 1.02))
    div, suf = (1e6, "M") if xr[1] >= 1e6 else (1e3, "k")
    fig, axs = plt.subplots(filas, cols, figsize=(9, 2.7 * filas), sharex=True, sharey=True, squeeze=False)
    for i, k in enumerate(orden):
        ax = axs[i // cols][i % cols]
        x = res[k][COL_U].to_numpy()
        c, bordes = np.histogram(x, bins=50, range=xr)
        cen = (bordes[:-1] + bordes[1:]) / 2
        ok = cen >= umbrales[k]
        ax.bar(cen[ok] / div, c[ok] / len(x) * 100, width=(bordes[1] - bordes[0]) / div * 0.95, color=PAL[i])
        ax.bar(cen[~ok] / div, c[~ok] / len(x) * 100, width=(bordes[1] - bordes[0]) / div * 0.95, color=PAL[i], alpha=0.3)
        ax.axvline(det_u[k] / div, color="#a86f00", lw=1.8)
        ax.axvline(umbrales[k] / div, color="#cc4444", lw=1.4, ls=":")
        ax.set_title(f"{k}  (éxito {np.mean(x >= umbrales[k]) * 100:.0f} %)", fontsize=9, loc="left")
        ax.set_xlim(xr[0] / div, xr[1] / div)
        _estilo(ax)
        if i // cols == filas - 1:
            ax.set_xlabel(f"Utilidad (US$ {suf})", fontsize=8)
        if i % cols == 0:
            ax.set_ylabel("% iteraciones", fontsize=8)
    for j in range(n, filas * cols):
        axs[j // cols][j % cols].axis("off")
    fig.text(0.99, 0.0, "Color intenso = cumple el criterio · amarillo = plan · rojo punteado = umbral", ha="right", fontsize=7.5, color="#555555")
    fig.tight_layout()
    return _png(fig)


def _g_zonas(orden, res, entradas):
    tz = {k: tabla_zonas(entradas[k], res[k]) for k in orden}
    n = len(orden)
    cols = min(n, 3)
    filas = math.ceil(n / cols)
    ymax = max(max(t["Utilidad plan (US$)"].max(), t["Utilidad media (US$)"].max()) for t in tz.values())
    ymin = min(0, min(min(t["Utilidad plan (US$)"].min(), t["Utilidad media (US$)"].min()) for t in tz.values()))
    div, suf = (1e6, "M") if ymax >= 1e6 else (1e3, "k")
    fig, axs = plt.subplots(filas, cols, figsize=(9, 2.7 * filas), sharey=True, squeeze=False)
    for i, k in enumerate(orden):
        ax = axs[i // cols][i % cols]
        t = tz[k]
        xs = np.arange(len(t))
        ax.bar(xs - 0.19, t["Utilidad plan (US$)"] / div, 0.36, color=PAL[i], alpha=0.35, label="Plan")
        ax.bar(xs + 0.19, t["Utilidad media (US$)"] / div, 0.36, color=PAL[i], label="Media simulada")
        ax.set_xticks(xs, t["Zona"], fontsize=8)
        ax.set_title(k, fontsize=9, loc="left")
        ax.set_ylim(ymin / div * 1.1, ymax / div * 1.15)
        _estilo(ax)
        if i % cols == 0:
            ax.set_ylabel(f"Utilidad (US$ {suf})", fontsize=8)
        if i == 0:
            ax.legend(frameon=False, fontsize=7.5)
    for j in range(n, filas * cols):
        axs[j // cols][j % cols].axis("off")
    fig.tight_layout()
    return _png(fig)


def _g_regla(cmin, cmax, puntos):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    c = np.linspace(40, 100, 241)
    ax.axvspan(cmin, cmax, color="#2a6fd1", alpha=0.10)
    ax.text(cmax - 0.5, 3, f"Rango del sorteo {cmin:.0f}–{cmax:.0f} %", ha="right", fontsize=7.5, color="#555555")
    ax.plot(c, factor_ley(c / 100) * 100, color="#222222", lw=2.6)
    vistos = set()
    for et, cz, ci in puntos:
        f = float(factor_ley(cz / 100)) * 100
        ax.plot([cz], [f], "o", color=PAL[ci], ms=7, mec="none", label=et.split(" · ")[0] if et.split(" · ")[0] not in vistos else None)
        vistos.add(et.split(" · ")[0])
    ax.set_xlabel("Certeza geo-operativa (%)", fontsize=8.5)
    ax.set_ylabel("Ley estimada (% de la ley base)", fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.set_xlim(40, 101)
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.1))
    _estilo(ax)
    fig.tight_layout()
    return _png(fig)


# ------------------------------------------------------------------------------ tablas
def _tabla(datos: list[list], anchos: list[float] | None = None, resaltar: list[tuple[int, int]] | None = None,
           alinear_der_desde: int = 1) -> Table:
    filas = []
    for r, fila in enumerate(datos):
        filas.append([Paragraph(escape(str(c)), ST["tch"] if r == 0 else ST["tc"]) for c in fila])
    t = Table(filas, colWidths=anchos, repeatRows=1)
    est = [
        ("BACKGROUND", (0, 0), (-1, 0), AZUL), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CLARO]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for (c, r) in resaltar or []:
        est.append(("BACKGROUND", (c, r), (c, r), VERDE))
    t.setStyle(TableStyle(est))
    return t


def _pie(proyecto: str, huella: str):
    def f(canvas, doc):
        canvas.saveState()
        canvas.setFont("DV", 7.5)
        canvas.setFillColor(GRIS)
        canvas.drawString(2 * cm, 1.2 * cm, f"Plataforma Montecarlo Minero · {proyecto} · huella {huella[:10]}")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Página {doc.page}")
        canvas.setStrokeColor(colors.HexColor("#cccccc"))
        canvas.line(2 * cm, 1.55 * cm, A4[0] - 2 * cm, 1.55 * cm)
        canvas.restoreState()
    return f


def _nombre_var(v: str) -> str:
    return v.replace("Certeza ", "Certeza de ")


# ------------------------------------------------------------------------------ informe
def pdf_bytes(*, entradas: dict[str, pd.DataFrame], supuestos: Supuestos, res: dict[str, pd.DataFrame],
              tot: dict[str, dict], det_u: dict[str, float], tabla: pd.DataFrame, modo: str, valor: float,
              umbrales: dict[str, float], ver: dict, huella: str, meta: dict) -> bytes:
    orden = list(res)
    t = tabla.set_index("Escenario")
    s = supuestos
    proyecto = meta.get("proyecto") or "Proyecto minero"
    fecha = meta.get("fecha") or date.today().strftime("%d/%m/%Y")
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2.1 * cm,
                            title=f"Informe de simulación de Montecarlo - {proyecto}", author=meta.get("elaborado_por") or "")
    el: list = []

    # ---- portada / encabezado
    el += [P("Informe técnico de simulación de Montecarlo", "t"),
           Paragraph(_t(f"**{proyecto}**" + (f" · {meta['empresa']}" if meta.get("empresa") else "")), ST["sub"]),
           Paragraph(_t(f"Evaluación probabilística de escenarios de producción · {fecha}"), ST["sub"]), Spacer(1, 8)]

    # ---- 1 resumen ejecutivo
    el.append(P("1. Resumen ejecutivo", "h1"))
    caja = [[Paragraph(_t(f"**{ver['titulo']}**"), ParagraphStyle("x", parent=ST["p"], fontSize=10.5, leading=14, alignment=0))]]
    for f in ver["frases"]:
        caja.append([Paragraph(_t("• " + f), ST["p"])])
    tb = Table(caja, colWidths=[W])
    tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CLARO), ("BOX", (0, 0), (-1, -1), 0.8, AZUL),
                            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    el += [tb, Spacer(1, 6)]
    div, suf = (1e6, "M") if max(det_u.values()) >= 1e6 else (1e3, "k")
    cab = ["Escenario", f"Plan (US$ {suf})", f"Media (US$ {suf})", "Cumpl. esperado", "Prob. de éxito",
           f"P5 – P95 (US$ {suf})", "Prob. de pérdida", "Certeza pond. (plan)"]
    filas = [cab]
    for n in orden:
        r = t.loc[n]
        filas.append([n, f"{det_u[n] / div:,.2f}", f"{r['Media (US$)'] / div:,.2f}", f"{r['Cumplimiento esperado']:.0%}",
                      f"{r['Prob. ≥ umbral']:.1%}", f"{r['P5'] / div:,.1f} – {r['P95'] / div:,.1f}",
                      f"{r['Prob. de pérdida (<0)']:.1%}", f"{tot[n]['Certeza geo-operativa (%)']:.0f} %"])
    mejor_i = 1 + orden.index(ver["recomendado"])
    el.append(_tabla(filas, [2.6 * cm, 1.9 * cm, 1.9 * cm, 2.2 * cm, 1.9 * cm, 2.6 * cm, 1.9 * cm, 2.0 * cm], resaltar=[(4, mejor_i)]))
    el.append(P("Criterio de éxito: " + ("alcanzar al menos el " + f"{valor:.0f} % de la utilidad planificada de cada escenario."
                if modo == "pct" else f"alcanzar al menos {_m(valor)} de utilidad.") + " En verde, el escenario de mayor probabilidad.", "c"))
    el.append(_g_exito(orden, tabla))

    # ---- 2 objetivo y método
    el.append(P("2. Objetivo y método", "h1"))
    el.append(P("El objetivo es comparar escenarios de producción minera bajo incertidumbre y determinar **cuál tiene mayor "
                "probabilidad de cumplir la utilidad planificada**. El plan (determinístico) se calcula con las fórmulas del modelo de "
                "evaluación por zonas, usando la certeza geo-operativa de la tabla. La simulación de Montecarlo repite el cálculo "
                f"{s.n_iter:,} veces sorteando, en cada iteración, **únicamente la certeza geo-operativa** de cada zona; el precio, "
                "las TMS, la ley base, la recuperación y los costos se mantienen fijos."))
    el.append(P("Por cada zona: ley estimada = ley base × (2 − 1/certeza); onzas = TMS × ley estimada × recuperación / 31,1035; "
                "valorizado = precio × onzas; costo = costo de explotación × TMS; utilidad = valorizado − costo. "
                "Los totales ponderan por las TMS de cada escenario."))
    el.append(P("**Regla certeza → ley.** La ley estimada es la ley base ajustada por la certeza: ley estimada = ley base + "
                "(ley base − ley base / certeza). Con 100 % de certeza se conserva la ley base; con 70 % se conserva el 57,1 %. "
                "Como el costo por tonelada no baja con la ley, la utilidad se mueve proporcionalmente más que la ley."))
    pts = [(f"{n} · {z['Zona']}", float(z[COL_CERTEZA]), i) for i, n in enumerate(orden) for _, z in entradas[n].iterrows()]
    el.append(_g_regla(s.certeza_min_pct, s.certeza_max_pct, pts))
    el.append(P("Figura: ley estimada según la certeza. Cada punto es una zona de un escenario con la certeza de la tabla; la banda es el rango en que se sortea la certeza.", "c"))

    el.append(P("Supuestos de la simulación", "h2"))
    sup = [["Supuesto", "Valor"], ["Iteraciones / semilla", f"{s.n_iter:,} / {s.semilla}"], ["Muestreo", "Latin Hypercube (números aleatorios comunes entre escenarios)"],
           ["Variable incierta", "Certeza geo-operativa de cada zona (única variable que se sortea)"],
           ["Datos fijos", "Precio, TMS, ley base, recuperación y costo de explotación (valores de la tabla)"]]
    det_cert = (f"{s.certeza_dist}: mínimo {s.certeza_min_pct:.0f} % · máximo {s.certeza_max_pct:.0f} %"
                + (" · más probable = certeza de la tabla de cada zona" if s.certeza_dist in ("Triangular", "PERT") else "")
                + (f" · desv. estándar {s.certeza_sd_pp:.1f} puntos" if s.certeza_dist == "Normal" else ""))
    sup += [["Certeza geo-operativa", det_cert + (", independiente por zona" if s.rho_zonas == 0 else f", correlación ρ = {s.rho_zonas:.2f}")],
            ["Misma zona entre escenarios", "Mismo sorteo" if s.compartir_geologia else "Sorteos independientes"],
            ["Criterio de éxito", ("≥ " + f"{valor:.0f} % del plan" if modo == "pct" else f"≥ {_m(valor)}")]]
    el.append(_tabla(sup, [5.5 * cm, W - 5.5 * cm]))

    # ---- 3 datos
    el.append(P("3. Datos de entrada por escenario", "h1"))
    for n in orden:
        d = tabla_determinista(entradas[n])
        filas = [["Zona", "TMS", "Ley base (g/t)", "Ley est. (g/t)", "Rec. (%)", "Precio (US$/oz)", "Costo (US$/t)", "Certeza (%)", "Onzas", "Utilidad plan (US$)"]]
        for (_, z), (_, dz) in zip(entradas[n].iterrows(), d.iterrows()):
            filas.append([z["Zona"], f"{z['TMS']:,.0f}", f"{z[COL_LEY]:.2f}", f"{dz[COL_LEY_EST]:.2f}", f"{z['Recuperación (%)']:.1f}",
                          f"{z['Precio (US$/oz)']:,.0f}", f"{z['Costo explotación (US$/t)']:.1f}", f"{z[COL_CERTEZA]:.0f}",
                          f"{dz['Onzas (oz)']:,.0f}", f"{dz['Utilidad (US$)']:,.0f}"])
        tt = tot[n]
        filas.append(["TOTAL", f"{tt['TMS']:,.0f}", f"{tt[COL_LEY]:.2f}", f"{tt[COL_LEY_EST]:.2f}", f"{tt['Recuperación (%)']:.1f}",
                      f"{tt['Precio (US$/oz)']:,.0f}", "", f"{tt[COL_CERTEZA]:.1f}", f"{tt['Onzas (oz)']:,.0f}", f"{tt['Utilidad (US$)']:,.0f}"])
        el.append(KeepTogether([P(n, "h2"), _tabla(filas, [1.7 * cm, 1.5 * cm, 1.5 * cm, 1.5 * cm, 1.2 * cm, 1.8 * cm, 1.5 * cm, 1.4 * cm, 1.6 * cm, 2.6 * cm])]))

    # ---- 4 determinístico
    el.append(P("4. Resultado determinístico (plan)", "h1"))
    el.append(P("Es el resultado de cada escenario con un solo valor por variable (la certeza geo-operativa de la tabla y el precio fijo). "
                "Indica la utilidad planificada, pero no la probabilidad de lograrla."))
    el.append(_g_determinista(orden, det_u, tot))

    # ---- 5 estocástico
    el.append(P("5. Resultado estocástico (Montecarlo)", "h1"))
    el.append(P("**Curva de cumplimiento.** Muestra, para cada porcentaje del plan, la probabilidad de alcanzarlo o superarlo. "
                "Si las curvas no se cruzan, el ranking no depende de qué tan exigente sea la meta.", "pk"))
    el.append(_g_curva(orden, res, det_u, modo, valor))
    el.append(P("**Distribución de la utilidad** (mismo eje para todos los escenarios).", "pk"))
    el.append(_g_histogramas(orden, res, det_u, umbrales))
    el.append(P("**Plan vs. media simulada por zona**: muestra en qué zonas se pierde el valor respecto al plan.", "pk"))
    el.append(_g_zonas(orden, res, entradas))
    filas = [["Escenario", f"P5 ({suf})", f"P10", "P50", "P90", f"P95", "Desv. est.", "Prob. mejor", "Onzas plan → media"]]
    for n in orden:
        r = t.loc[n]
        filas.append([n, f"{r['P5'] / div:,.2f}", f"{r['P10'] / div:,.2f}", f"{r['P50'] / div:,.2f}", f"{r['P90'] / div:,.2f}",
                      f"{r['P95'] / div:,.2f}", f"{r['Desv. estándar (US$)'] / div:,.2f}", f"{r['Prob. de ser el mejor']:.1%}",
                      f"{tot[n]['Onzas (oz)']:,.0f} → {r['Onzas medias (oz)']:,.0f}"])
    el.append(_tabla(filas, [2.6 * cm, 1.5 * cm, 1.5 * cm, 1.5 * cm, 1.5 * cm, 1.5 * cm, 1.8 * cm, 1.9 * cm, 3.4 * cm]))
    el.append(P(f"Valores en US$ {suf}. Error estándar de una probabilidad con {s.n_iter:,} iteraciones: ≤ {50 / math.sqrt(s.n_iter):.1f} puntos.", "c"))

    # ---- 6 análisis técnico
    el.append(P("6. Análisis técnico", "h1"))
    for f in ver["frases"][:3]:
        el.append(B(f))
    for n in orden:
        r = t.loc[n]
        tz = tabla_zonas(entradas[n], res[n])
        tz["perdida"] = tz["Utilidad plan (US$)"] - tz["Utilidad media (US$)"]
        peor = tz.sort_values("perdida", ascending=False).iloc[0]
        sens = sensibilidad(res[n])
        motor = ""
        if not sens.empty:
            top = sens.iloc[-1]
            motor = f" La variable que más mueve su utilidad es **{_nombre_var(top['Variable'])}** (correlación {top['Correlación']:+.2f})."
        el.append(B(
            f"**{n}:** plan {_m(det_u[n])}, media simulada {_m(r['Media (US$)'])} ({r['Cumplimiento esperado']:.0%} del plan); "
            f"90 % de los resultados entre {_m(r['P5'])} y {_m(r['P95'])}; probabilidad de éxito {r['Prob. ≥ umbral']:.1%}. "
            f"La mayor diferencia respecto al plan está en **{peor['Zona']}** (certeza en el plan {peor['Certeza plan (%)']:.0f} %, "
            f"simulada {peor['Certeza simulada media (%)']:.1f} %; ley estimada {peor['Ley estimada plan (g/t)']:.2f} → "
            f"{peor['Ley estimada media (g/t)']:.2f} g/t).{motor}"))
    el.append(P("Interpretación", "h2"))
    el.append(P("La certeza geo-operativa define la ley estimada y, con ella, las onzas y la utilidad. Un escenario puede tener el mayor "
                "plan y, a la vez, una probabilidad de cumplirlo distinta de la de otro, según cuánto tonelaje asigna a cada zona. "
                "La probabilidad de éxito y el cumplimiento esperado miden el riesgo que el plan determinístico no muestra."))

    # ---- 7 limitaciones
    el.append(P("7. Limitaciones y recomendaciones", "h1"))
    for a in ver["avisos"]:
        el.append(B(a))
    for x in ["La utilidad corresponde a un único periodo; no incluye valor del dinero en el tiempo (VAN), capex ni cronograma de producción.",
              "Solo la certeza geo-operativa es variable incierta; precio, tonelaje, ley base, recuperación y costos se mantienen fijos.",
              "Recomendado: validar el rango de la certeza geo-operativa y su distribución con datos de perforación y conciliación mina-planta; "
              "ampliar a categorías de recurso (medido, indicado, inferido) por zona.",
              "Validar los resultados con un segundo software o hoja de cálculo independiente antes de usarlos en una decisión de inversión."]:
        el.append(B(x))

    # ---- 8 reproducibilidad
    el.append(P("8. Reproducibilidad", "h1"))
    el.append(P(f"Iteraciones: {s.n_iter:,} · semilla: {s.semilla} · huella de datos y supuestos (MD5): {huella}. "
                "Con los mismos datos, supuestos y semilla, la plataforma reproduce estos resultados exactamente."))

    # ---- 9 firmas
    def celda_firma(rol, nombre, cargo, img):
        partes = [Paragraph(f"<b>{escape(rol)}</b>", ST["tc"]), Spacer(1, 2)]
        if img:
            try:
                im = Image(io.BytesIO(img))
                ratio = im.imageHeight / im.imageWidth
                ancho = min(5.2 * cm, 2.0 * cm / ratio)
                partes.append(Image(io.BytesIO(img), width=ancho, height=ancho * ratio))
            except Exception:  # noqa: BLE001
                partes.append(Spacer(1, 1.8 * cm))
        else:
            partes.append(Spacer(1, 1.8 * cm))
        partes += [Paragraph("_" * 38, ST["tc"]), Paragraph(escape(nombre or "Nombre y apellidos"), ST["tc"]),
                   Paragraph(escape(cargo or "Cargo"), ST["tc"]), Paragraph(f"Fecha: {fecha}", ST["tc"])]
        return partes

    firmas = Table([[celda_firma("Elaborado por", meta.get("elaborado_por"), meta.get("cargo"), meta.get("firma1")),
                     celda_firma("Revisado y aprobado por", meta.get("revisado_por"), meta.get("cargo_rev"), meta.get("firma2"))]],
                   colWidths=[W / 2, W / 2])
    firmas.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6)]))
    el.append(KeepTogether([P("9. Firmas", "h1"), Spacer(1, 6), firmas]))

    pie = _pie(proyecto, huella)
    doc.build(el, onFirstPage=pie, onLaterPages=pie)
    return buf.getvalue()
