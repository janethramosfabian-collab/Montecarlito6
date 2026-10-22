import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from core import Supuestos, simular, totales_deterministas, tabla_determinista, factor_ley, ley_estimada
from core.analisis import tabla_zonas, veredicto
from core.distribuciones import CATALOGO, congelar
from core.ejemplo import escenarios_ejemplo
from core.estadistica import curva_cumplimiento, prob_mejor, resumen, sensibilidad

E = escenarios_ejemplo()


def test_determinista_coincide_con_excel():
    esperado = {"Escenario 1": 18569131.33, "Escenario 2": 22207576.66, "Escenario 3": 22747207.60}
    for k, v in esperado.items():
        assert totales_deterministas(E[k])["Utilidad (US$)"] == pytest.approx(v, abs=0.01)


def test_onzas_ley_estimada_y_certeza_como_en_el_excel():
    t = tabla_determinista(E["Escenario 1"])
    assert t["Ley estimada (g/t)"].round(3).tolist() == [4.0, 4.318, 5.814]
    assert t["Onzas (oz)"].round(0).tolist() == [611, 1263, 2608]
    tot3 = totales_deterministas(E["Escenario 3"])
    assert tot3["Onzas (oz)"] == pytest.approx(5468.27, abs=0.01)
    assert tot3["Certeza geo-operativa (%)"] == pytest.approx(82.6)
    assert totales_deterministas(E["Escenario 1"])["Certeza geo-operativa (%)"] == pytest.approx(89.5)
    assert totales_deterministas(E["Escenario 2"])["Certeza geo-operativa (%)"] == pytest.approx(85.9)


def test_totales_ponderados_por_tms_propio():
    # La fila Total de la ley estimada usa las TMS del propio escenario (en el Excel, E2 y E3 usaban las del E1 por error)
    assert totales_deterministas(E["Escenario 2"])["Ley estimada (g/t)"] == pytest.approx(5.9298, abs=1e-3)
    assert totales_deterministas(E["Escenario 3"])["Ley estimada (g/t)"] == pytest.approx(6.0821, abs=1e-3)
    assert totales_deterministas(E["Escenario 3"])["Ley base (g/t)"] == pytest.approx(7.4)


def test_regla_del_excel():
    # ley estimada = ley base + (ley base − ley base / certeza)
    assert float(ley_estimada(7.0, 0.70)) == pytest.approx(4.0)
    assert float(factor_ley(1.0)) == pytest.approx(1.0)
    assert float(factor_ley(0.5)) == pytest.approx(0.0)
    assert float(factor_ley(0.3)) == 0.0                    # nunca negativa
    for c in (0.7, 0.88, 0.97):
        assert float(factor_ley(c)) == pytest.approx(1 + (1 - 1 / c))


def test_solo_varia_la_certeza():
    r = simular(E, Supuestos(n_iter=2000))
    for k, df in r.items():
        assert not any(c.startswith(("Factor precio", "Factor ley")) for c in df.columns)
        assert df["Utilidad (US$)"].std() > 0
    # con rango degenerado (mín = máx = certeza) todas las iteraciones son iguales
    s = Supuestos(n_iter=200, certeza_min_pct=88, certeza_max_pct=88)
    for k, df in simular({"X": E["Escenario 1"].assign(**{"Certeza geo-operativa (%)": 88.0})}, s).items():
        assert df["Utilidad (US$)"].std() < 1e-6


def test_certeza_dentro_del_rango_70_97():
    for d in ["Triangular", "PERT", "Uniforme", "Normal"]:
        r = simular(E, Supuestos(n_iter=5000, certeza_dist=d))["Escenario 1"]
        for z in ("Zona 1", "Zona 2", "Zona 3"):
            assert r[f"Certeza {z}"].between(70 - 1e-9, 97 + 1e-9).all(), (d, z)
    # la media de la Uniforme es el centro del rango
    r = simular(E, Supuestos(n_iter=20000, certeza_dist="Uniforme"))["Escenario 1"]
    assert r["Certeza Zona 2"].mean() == pytest.approx(83.5, abs=0.2)


def test_triangular_usa_la_certeza_de_la_tabla_como_valor_mas_probable():
    r = simular(E, Supuestos(n_iter=40000, certeza_dist="Triangular"))["Escenario 1"]
    # Triangular(70, moda, 97): media = (70 + moda + 97) / 3
    assert r["Certeza Zona 1"].mean() == pytest.approx((70 + 70 + 97) / 3, abs=0.2)
    assert r["Certeza Zona 2"].mean() == pytest.approx((70 + 88 + 97) / 3, abs=0.2)
    assert r["Certeza Zona 3"].mean() == pytest.approx((70 + 97 + 97) / 3, abs=0.2)


def test_iteraciones_soportadas():
    for n in (20000, 30000, 50000, 100000):
        r = simular({"Escenario 1": E["Escenario 1"]}, Supuestos(n_iter=n))
        assert len(r["Escenario 1"]) == n


def test_reproducible_con_semilla():
    a = simular(E, Supuestos(n_iter=500, semilla=3))["Escenario 1"]["Utilidad (US$)"]
    b = simular(E, Supuestos(n_iter=500, semilla=3))["Escenario 1"]["Utilidad (US$)"]
    c = simular(E, Supuestos(n_iter=500, semilla=4))["Escenario 1"]["Utilidad (US$)"]
    assert (a == b).all() and not (a == c).all()


def test_certeza_independiente_por_zona_y_compartida_entre_escenarios():
    r = simular(E, Supuestos(n_iter=5000, certeza_dist="Uniforme", rho_zonas=0.0))
    a = r["Escenario 1"]
    assert abs(np.corrcoef(a["Certeza Zona 1"], a["Certeza Zona 2"])[0, 1]) < 0.05      # independientes entre zonas
    r2 = simular(E, Supuestos(n_iter=300, compartir_geologia=True, certeza_dist="Uniforme"))
    assert np.allclose(r2["Escenario 1"]["Certeza Zona 1"], r2["Escenario 2"]["Certeza Zona 1"])   # misma zona, mismo sorteo
    r3 = simular(E, Supuestos(n_iter=300, compartir_geologia=False, certeza_dist="Uniforme"))
    assert not np.allclose(r3["Escenario 1"]["Certeza Zona 1"], r3["Escenario 2"]["Certeza Zona 1"])


def test_correlacion_entre_zonas():
    r = simular(E, Supuestos(n_iter=20000, rho_zonas=0.8, certeza_dist="Uniforme"))["Escenario 1"]
    assert np.corrcoef(r["Certeza Zona 1"], r["Certeza Zona 3"])[0, 1] > 0.6


def test_utilidad_simulada_cerca_del_plan_y_ranking():
    r = simular(E, Supuestos(n_iter=20000, semilla=5))
    det = {k: totales_deterministas(v)["Utilidad (US$)"] for k, v in E.items()}
    for k, df in r.items():
        assert 0.85 < df["Utilidad (US$)"].mean() / det[k] < 1.15
    p = {k: curva_cumplimiento(df["Utilidad (US$)"].to_numpy(), det[k], [100])[0] for k, df in r.items()}
    assert max(p, key=p.get) == "Escenario 3"


def test_prob_mejor_tornado_y_veredicto():
    r = simular(E, Supuestos(n_iter=3000))
    assert sum(prob_mejor(r).values()) == pytest.approx(1.0)
    t = sensibilidad(r["Escenario 1"])
    assert set(t["Variable"]) == {"Certeza Zona 1", "Certeza Zona 2", "Certeza Zona 3"}
    det = {k: totales_deterministas(v)["Utilidad (US$)"] for k, v in E.items()}
    tabla = resumen(r, det, "pct", 100)
    assert "Certeza simulada media (%)" in tabla.columns
    ver = veredicto(tabla, "pct", 100, {k: totales_deterministas(v)["Certeza geo-operativa (%)"] for k, v in E.items()}, 3000)
    assert ver["recomendado"] in E and ver["frases"] and ver["avisos"]
    tz = tabla_zonas(E["Escenario 3"], r["Escenario 3"])
    assert (tz["Ley estimada plan (g/t)"] > 0).all()
    assert tz["Utilidad plan (US$)"].sum() == pytest.approx(det["Escenario 3"], rel=1e-9)


def test_catalogo_de_distribuciones():
    for nombre, info in CATALOGO.items():
        assert info["coloquial"] and info["tecnica"] and info["formulas"]
        if info["params"]:
            d = congelar(nombre, **{k: df for k, _, _, _, df, _ in info["params"]})
            assert np.isfinite(d.mean()) and d.ppf(0.05) < d.ppf(0.95)
    assert congelar("Lognormal", cv=10.0).mean() == pytest.approx(0.0, abs=1e-6)


def test_exportar_excel_y_pdf():
    import io, openpyxl
    from core.exportar import excel_bytes
    from core.informe_pdf import pdf_bytes
    s = Supuestos(n_iter=2000)
    r = simular(E, s)
    det = {k: totales_deterministas(v)["Utilidad (US$)"] for k, v in E.items()}
    tot = {k: totales_deterministas(v) for k, v in E.items()}
    tabla = resumen(r, det, "pct", 100)
    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes(E, s, tabla, r)))
    assert {"Supuestos", "Determinístico", "Resumen estocástico", "Sim Escenario 1"} <= set(wb.sheetnames)
    valores = [c.value for fila in wb["Determinístico"].iter_rows() for c in fila if isinstance(c.value, float)]
    assert any(abs(v - 18569131.33) < 0.01 for v in valores)
    ver = veredicto(tabla, "pct", 100, {k: v["Certeza geo-operativa (%)"] for k, v in tot.items()}, 2000)
    um = {k: det[k] * 1.0 for k in det}
    pdf = pdf_bytes(entradas=E, supuestos=s, res=r, tot=tot, det_u=det, tabla=tabla, modo="pct", valor=100, umbrales=um, ver=ver,
                    huella="abc123", meta={"proyecto": "Prueba", "elaborado_por": "Ana", "cargo": "Ing."})
    assert pdf[:4] == b"%PDF" and len(pdf) > 30000
