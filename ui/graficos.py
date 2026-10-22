"""Gráficos Plotly: un eje por panel, color fijo por escenario, rejilla discreta."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Paleta categórica en orden fijo (tema oscuro). Los 3 primeros escenarios validan entre sí;
# cada escenario conserva SIEMPRE su color en toda la app.
PALETA = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
TEXTO = "#c3c2b7"
TEXTO_FUERTE = "#f0efec"
REJILLA = "rgba(240,239,236,0.10)"
POSITIVO, NEGATIVO = "#3987e5", "#e66767"


def color_de(i: int) -> str:
    return PALETA[i % len(PALETA)]


def rgba(hex_: str, alpha: float) -> str:
    h = hex_.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def escala(valores) -> tuple[float, str]:
    """Divisor y sufijo legibles para US$ (M, k o unidades)."""
    m = float(np.nanmax(np.abs(valores))) if len(valores) else 0.0
    if m >= 1e6:
        return 1e6, "M"
    if m >= 1e3:
        return 1e3, "k"
    return 1.0, ""


def _base(fig: go.Figure, alto: int = 380, titulo: str | None = None) -> go.Figure:
    fig.update_layout(
        title=dict(text=titulo, x=0.0, font=dict(size=15, color=TEXTO_FUERTE)) if titulo else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXTO, size=13),
        height=alto,
        margin=dict(l=10, r=10, t=50 if titulo else 20, b=10),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(color=TEXTO)),
        hoverlabel=dict(bgcolor="#2a2a28", font=dict(color=TEXTO_FUERTE)),
    )
    fig.update_xaxes(gridcolor=REJILLA, zerolinecolor="rgba(240,239,236,0.25)", linecolor=REJILLA)
    fig.update_yaxes(gridcolor=REJILLA, zerolinecolor="rgba(240,239,236,0.25)", linecolor=REJILLA)
    return fig


# ----------------------------------------------------------------------------------
def fig_determinista(nombres, utilidad, onzas) -> go.Figure:
    """Caso determinístico: utilidad (arriba) y onzas recuperables (abajo)."""
    div, suf = escala(utilidad)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, row_heights=[0.6, 0.4])
    for i, n in enumerate(nombres):
        fig.add_bar(
            x=[n], y=[utilidad[i] / div], name=n, marker_color=color_de(i), width=0.38, row=1, col=1,
            text=[f"{utilidad[i] / div:,.2f}"], textposition="outside", textfont=dict(color=TEXTO_FUERTE),
            hovertemplate=f"<b>{n}</b><br>Utilidad: US$ %{{y:,.3f}} {suf}<extra></extra>",
        )
        fig.add_bar(
            x=[n], y=[onzas[i]], name=n, marker_color=color_de(i), width=0.38, row=2, col=1, showlegend=False,
            text=[f"{onzas[i]:,.0f}"], textposition="outside", textfont=dict(color=TEXTO_FUERTE),
            hovertemplate=f"<b>{n}</b><br>Onzas: %{{y:,.0f}} oz<extra></extra>",
        )
    fig.update_yaxes(title_text=f"Utilidad (US$ {suf})".replace(" )", ")"), rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="Onzas recuperables (oz)", rangemode="tozero", row=2, col=1)
    ymax = max(utilidad) / div
    fig.update_yaxes(range=[0, ymax * 1.18], row=1, col=1)
    fig.update_yaxes(range=[0, max(onzas) * 1.25], row=2, col=1)
    fig.update_layout(barmode="overlay")
    return _base(fig, 470, "Caso determinístico: un solo resultado por escenario")


def fig_probabilidad(nombres, utilidad_det, p5, p95, probs, umbrales) -> go.Figure:
    """Toque estocástico: utilidad con su rango P5–P95 (arriba) y probabilidad de cumplir el umbral (abajo)."""
    div, suf = escala(utilidad_det)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, row_heights=[0.55, 0.45])
    for i, n in enumerate(nombres):
        c = color_de(i)
        fig.add_bar(
            x=[n], y=[utilidad_det[i] / div], name=n, marker_color=rgba(c, 0.45), width=0.38, row=1, col=1,
            hovertemplate=f"<b>{n}</b><br>Plan: US$ %{{y:,.3f}} {suf}<extra></extra>",
        )
        fig.add_scatter(
            x=[n, n], y=[p5[i] / div, p95[i] / div], mode="lines+markers", name=n, showlegend=False,
            line=dict(color=c, width=3), marker=dict(size=9, color=c, line=dict(color="#1a1a19", width=2)),
            hovertemplate=f"<b>{n}</b><br>%{{y:,.3f}} {suf} (P5 → P95)<extra></extra>", row=1, col=1,
        )
        fig.add_bar(
            x=[n], y=[probs[i] * 100], name=n, marker_color=c, width=0.38, row=2, col=1, showlegend=False,
            text=[f"{probs[i] * 100:,.1f}%"], textposition="outside", textfont=dict(color=TEXTO_FUERTE),
            hovertemplate=(f"<b>{n}</b><br>Prob. de utilidad ≥ US$ {umbrales[i] / div:,.2f} {suf}: "
                           f"%{{y:,.1f}}%<extra></extra>"),
        )
    fig.update_yaxes(title_text=f"Utilidad (US$ {suf})", row=1, col=1)
    fig.update_yaxes(title_text="Probabilidad de cumplir (%)", range=[0, 118], row=2, col=1)
    fig.update_layout(barmode="overlay")
    return _base(fig, 470, "Con incertidumbre: rango P5–P95 y probabilidad de cumplir")


def fig_scurves(resultados: dict, columna: str, umbrales: dict[str, float], orden: list[str]) -> go.Figure:
    """Curvas de excedencia P(X ≥ x). El rombo marca la probabilidad en el umbral de cada escenario."""
    todo = np.concatenate([resultados[n][columna].to_numpy() for n in orden])
    div, suf = escala(todo)
    fig = go.Figure()
    for i, n in enumerate(orden):
        x = np.sort(resultados[n][columna].to_numpy())
        p = 1.0 - np.arange(len(x)) / len(x)
        paso = max(1, len(x) // 400)
        fig.add_scatter(
            x=x[::paso] / div, y=p[::paso] * 100, mode="lines", name=n, line=dict(color=color_de(i), width=2.5),
            hovertemplate=f"<b>{n}</b><br>US$ %{{x:,.2f}} {suf}<br>Prob. de superarlo: %{{y:,.1f}}%<extra></extra>",
        )
        um = umbrales[n]
        pu = float(np.mean(x >= um)) * 100
        fig.add_scatter(
            x=[um / div], y=[pu], mode="markers", showlegend=False,
            marker=dict(symbol="diamond", size=11, color=color_de(i), line=dict(color="#1a1a19", width=2)),
            hovertemplate=f"<b>{n}</b> · umbral US$ %{{x:,.2f}} {suf}<br>Prob. ≥ umbral: %{{y:,.1f}}%<extra></extra>",
        )
    fig.update_xaxes(title_text=f"Utilidad (US$ {suf})")
    fig.update_yaxes(title_text="Probabilidad de superar el valor (%)", range=[0, 100])
    return _base(fig, 400, "Curvas de excedencia (◆ = umbral de cada escenario)")


def fig_histograma(x: np.ndarray, rango: tuple[float, float], det: float, color: str, titulo: str,
                   etiqueta: str = "Utilidad", xrange: tuple[float, float] | None = None,
                   umbral: float | None = None, alto: int = 360) -> go.Figure:
    """Histograma con las barras dentro del rango resaltadas (la 'barra de probabilidad').
    `xrange` fija el mismo eje en todos los escenarios (para compararlos a ojo)."""
    div, suf = escala(x if xrange is None else np.array(xrange))
    cuentas, bordes = np.histogram(x, bins=60, range=xrange)
    centros = (bordes[:-1] + bordes[1:]) / 2
    dentro = (centros >= rango[0]) & (centros <= rango[1])
    colores = [rgba(color, 0.95) if d else rgba(color, 0.25) for d in dentro]
    fig = go.Figure(go.Bar(
        x=centros / div, y=cuentas / len(x) * 100, marker_color=colores, width=(bordes[1] - bordes[0]) / div * 0.96,
        hovertemplate=f"US$ %{{x:,.2f}} {suf}<br>%{{y:,.2f}}% de las iteraciones<extra></extra>", showlegend=False,
    ))
    for v in rango:
        fig.add_vline(x=v / div, line_dash="dash", line_color=TEXTO_FUERTE, line_width=1.5)
    fig.add_vline(x=det / div, line_color="#c98500", line_width=2.5, annotation_text="Plan",
                  annotation_position="top", annotation_font=dict(color="#c98500"))
    if umbral is not None:
        fig.add_vline(x=umbral / div, line_color="#e66767", line_width=2, line_dash="dot", annotation_text="Umbral",
                      annotation_position="bottom", annotation_font=dict(color="#e66767"))
    if xrange is not None:
        fig.update_xaxes(range=[xrange[0] / div, xrange[1] / div])
    fig.update_xaxes(title_text=f"{etiqueta} (US$ {suf})")
    fig.update_yaxes(title_text="% de iteraciones")
    return _base(fig, alto, titulo)


def fig_tornado(df, titulo: str = "Sensibilidad (correlación de rangos de Spearman)") -> go.Figure:
    nombres = []
    for v in df["Variable"]:
        v = v.replace("Certeza ", "Certeza · ")
        nombres.append(v)
    fig = go.Figure(go.Bar(
        x=df["Correlación"], y=nombres, orientation="h", width=0.55,
        marker_color=[POSITIVO if v >= 0 else NEGATIVO for v in df["Correlación"]],
        text=[f"{v:+.2f}" for v in df["Correlación"]], textposition="outside", textfont=dict(color=TEXTO_FUERTE),
        hovertemplate="%{y}<br>Correlación con la utilidad: %{x:+.3f}<extra></extra>",
    ))
    fig.update_xaxes(range=[-1.1, 1.1], title_text="Correlación con la utilidad")
    return _base(fig, max(260, 60 + 42 * len(df)), titulo)


def fig_curva_cumplimiento(res: dict, det: dict, orden: list[str], x_actual: float | None = None) -> go.Figure:
    """Probabilidad de alcanzar al menos X % del plan, para todo X. La línea vertical es el X elegido."""
    from core.estadistica import COL_U, curva_cumplimiento
    xs = np.arange(0, 121, 1.0)
    fig = go.Figure()
    for i, n in enumerate(orden):
        p = curva_cumplimiento(res[n][COL_U].to_numpy(), det[n], xs) * 100
        fig.add_scatter(x=xs, y=p, mode="lines", name=n, line=dict(color=color_de(i), width=2.8),
                        hovertemplate=f"<b>{n}</b><br>Alcanzar ≥ %{{x:.0f}}% del plan: %{{y:.1f}}%<extra></extra>")
        if x_actual is not None:
            py = float(curva_cumplimiento(res[n][COL_U].to_numpy(), det[n], [x_actual])[0] * 100)
            fig.add_scatter(x=[x_actual], y=[py], mode="markers+text", showlegend=False, text=[f"{py:.0f}%"],
                            textposition="middle right", textfont=dict(color=color_de(i), size=13),
                            marker=dict(size=11, color=color_de(i), line=dict(color="#1a1a19", width=2)),
                            hoverinfo="skip")
    if x_actual is not None:
        fig.add_vline(x=x_actual, line_dash="dash", line_color=TEXTO_FUERTE, line_width=1.2)
    fig.add_vline(x=100, line_color="#c98500", line_width=2, annotation_text="Plan (100 %)",
                  annotation_position="top", annotation_font=dict(color="#c98500"))
    fig.update_xaxes(title_text="Utilidad alcanzada (% del plan determinístico)", range=[0, 120])
    fig.update_yaxes(title_text="Probabilidad de alcanzarla o superarla (%)", range=[0, 103])
    return _base(fig, 420, "¿Qué tan probable es cumplir el plan? (curva por escenario)")


def fig_ley_certeza(cmin: float, cmax: float, puntos: list[tuple[str, float, str]]) -> go.Figure:
    """Regla del Excel: ley estimada = ley base × (2 − 1/certeza). `puntos` = (etiqueta, certeza %, color).
    La banda sombreada es el rango [cmin, cmax] en que se sortea la certeza."""
    c = np.linspace(40, 100, 241)
    f = np.clip(2 - 1 / (c / 100), 0, None) * 100
    fig = go.Figure()
    fig.add_vrect(x0=cmin, x1=cmax, fillcolor=rgba(PALETA[0], 0.12), line_width=0,
                  annotation_text=f"Rango del sorteo {cmin:.0f}–{cmax:.0f} %", annotation_position="bottom right",
                  annotation_font=dict(color=TEXTO, size=11))
    fig.add_scatter(x=c, y=f, mode="lines", name="Regla", showlegend=False, line=dict(color=TEXTO_FUERTE, width=3),
                    hovertemplate="Certeza %{x:.0f}%<br>Ley estimada = %{y:.1f}% de la ley base<extra></extra>")
    vistos: set[str] = set()
    for et, cz, col in puntos:
        esc = et.split(" · ")[0] if " · " in et else None
        mostrar = esc is not None and esc not in vistos
        if esc:
            vistos.add(esc)
        fy = max(2 - 1 / (cz / 100), 0) * 100
        fig.add_scatter(x=[cz], y=[fy], mode="markers", name=esc or et, showlegend=mostrar, legendgroup=esc,
                        marker=dict(size=12, color=col, line=dict(color="#1a1a19", width=2)),
                        hovertemplate=f"<b>{et}</b><br>Certeza {cz:.0f}% → ley estimada al {fy:.1f}% de la ley base<extra></extra>")
    fig.update_xaxes(title_text="Certeza geo-operativa (%)", range=[40, 101])
    fig.update_yaxes(title_text="Ley estimada (% de la ley base)", range=[0, 105])
    return _base(fig, 380, "Cuánta ley base se conserva según la certeza: 2 − 1/certeza")


def fig_certeza(orden: list[str], plan: list[float], media: list[float], p5: list[float], p95: list[float]) -> go.Figure:
    """Certeza geo-operativa ponderada de cada escenario: plan (rombo) y rango simulado P5–P95 con su media."""
    fig = go.Figure()
    for i, n in enumerate(orden):
        c = color_de(i)
        fig.add_scatter(x=[p5[i], p95[i]], y=[n, n], mode="lines", showlegend=False, line=dict(color=rgba(c, 0.55), width=9),
                        hovertemplate=f"<b>{n}</b><br>P5 – P95: {p5[i]:.1f}% – {p95[i]:.1f}%<extra></extra>")
        fig.add_scatter(x=[media[i]], y=[n], mode="markers", name="Media simulada", showlegend=(i == 0),
                        marker=dict(size=13, color=c, line=dict(color="#1a1a19", width=2)),
                        hovertemplate=f"<b>{n}</b><br>Media simulada: %{{x:.1f}}%<extra></extra>")
        fig.add_scatter(x=[plan[i]], y=[n], mode="markers", name="Plan (determinístico)", showlegend=(i == 0),
                        marker=dict(size=13, symbol="diamond-open", color=TEXTO_FUERTE, line=dict(color=TEXTO_FUERTE, width=2.5)),
                        hovertemplate=f"<b>{n}</b><br>Plan: %{{x:.1f}}%<extra></extra>")
    fig.update_xaxes(title_text="Certeza geo-operativa ponderada por TMS (%)")
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 300, "Certeza geo-operativa del escenario: plan vs. simulada (rango P5–P95)")


def fig_distribucion(nombre: str, d, color: str = PALETA[0], unidad: str = "Variación (%)") -> go.Figure:
    """Densidad (arriba) y acumulada (abajo) de una distribución continua de scipy."""
    lo, hi = float(d.ppf(0.0005)), float(d.ppf(0.9995))
    x = np.linspace(lo, hi, 400)
    p5, p50, p95 = (float(d.ppf(q)) for q in (0.05, 0.5, 0.95))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, row_heights=[0.6, 0.4])
    fig.add_scatter(x=x, y=d.pdf(x), mode="lines", line=dict(color=color, width=3), showlegend=False,
                    hovertemplate="%{x:.1f}<br>densidad %{y:.4f}<extra></extra>", row=1, col=1)
    xs = x[(x >= p5) & (x <= p95)]
    fig.add_scatter(x=xs, y=d.pdf(xs), mode="lines", fill="tozeroy", fillcolor=rgba(color, 0.25),
                    line=dict(width=0), showlegend=False, hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=x, y=d.cdf(x) * 100, mode="lines", line=dict(color=color, width=3), showlegend=False,
                    hovertemplate="%{x:.1f}<br>acumulada %{y:.1f}%<extra></extra>", row=2, col=1)
    for v, t in [(p5, "P5"), (p50, "P50"), (p95, "P95")]:
        fig.add_vline(x=v, line_dash="dot", line_color=TEXTO, line_width=1, row=1, col=1)
        fig.add_vline(x=v, line_dash="dot", line_color=TEXTO, line_width=1, row=2, col=1)
        fig.add_annotation(x=v, y=1.0, yref="y domain", text=t, showarrow=False, yshift=10, font=dict(color=TEXTO, size=12),
                           row=1, col=1)
    fig.update_yaxes(title_text="Densidad", showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="Acumulada (%)", range=[0, 102], row=2, col=1)
    fig.update_xaxes(title_text=unidad, row=2, col=1)
    return _base(fig, 400, f"{nombre}: forma (arriba) y probabilidad acumulada (abajo)")


def fig_convergencia(x: np.ndarray, div: float, suf: str) -> go.Figure:
    """Media acumulada con banda ±2 errores estándar: por qué más iteraciones dan resultados más estables."""
    n = np.arange(1, len(x) + 1)
    media = np.cumsum(x) / n
    sd = np.sqrt(np.maximum(np.cumsum(x**2) / n - media**2, 0))
    se = sd / np.sqrt(n)
    idx = np.unique(np.round(np.logspace(1, np.log10(len(x)), 300)).astype(int)) - 1
    fig = go.Figure()
    fig.add_scatter(x=n[idx], y=(media[idx] + 2 * se[idx]) / div, mode="lines", line=dict(width=0), showlegend=False,
                    hoverinfo="skip")
    fig.add_scatter(x=n[idx], y=(media[idx] - 2 * se[idx]) / div, mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor=rgba(PALETA[0], 0.25), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=n[idx], y=media[idx] / div, mode="lines", line=dict(color=PALETA[0], width=2.8), showlegend=False,
                    hovertemplate="%{x:,.0f} iteraciones<br>media %{y:,.3f} " + suf + "<extra></extra>")
    fig.update_xaxes(title_text="Número de iteraciones (escala log)", type="log")
    fig.update_yaxes(title_text=f"Utilidad media acumulada (US$ {suf})")
    return _base(fig, 340, "La media se estabiliza al aumentar las iteraciones (banda = ±2 errores estándar)")


def fig_zonas(nombres_zonas: list[str], plan: list[float], media: list[float], color: str, div: float, suf: str,
              yrange: tuple[float, float] | None = None, titulo: str = "") -> go.Figure:
    """Utilidad planificada vs. utilidad media simulada de cada zona."""
    fig = go.Figure()
    fig.add_bar(x=nombres_zonas, y=[v / div for v in plan], name="Plan", marker_color=rgba(color, 0.35), width=0.36,
                text=[f"{v / div:,.2f}" for v in plan], textposition="outside", textfont=dict(color=TEXTO))
    fig.add_bar(x=nombres_zonas, y=[v / div for v in media], name="Media simulada", marker_color=color, width=0.36,
                text=[f"{v / div:,.2f}" for v in media], textposition="outside", textfont=dict(color=TEXTO_FUERTE))
    fig.update_layout(barmode="group", bargap=0.35)
    if yrange:
        fig.update_yaxes(range=[yrange[0] / div, yrange[1] / div])
    fig.update_yaxes(title_text=f"Utilidad (US$ {suf})")
    return _base(fig, 330, titulo)
