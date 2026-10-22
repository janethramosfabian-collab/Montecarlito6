import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)
from streamlit.testing.v1 import AppTest

def test_app_arranca_simula_y_no_falla():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not at.exception, at.exception
    boton = [b for b in at.sidebar.button if "Simular" in b.label][0]
    boton.click().run()
    assert not at.exception, at.exception
    assert at.session_state["sim"]["resultados"].keys() == {"Escenario 1", "Escenario 2", "Escenario 3"}
    assert len(at.session_state["sim"]["resultados"]["Escenario 1"]) == 20000
    # slider de umbral responde sin romper
    for sl in at.slider:
        if sl.key == "u_pct":
            sl.set_value(90).run()
    assert not at.exception, at.exception
    assert len(at.metric) > 10
    assert at.session_state["criterio"] == ("pct", 90.0)
    assert at.session_state["veredicto"]["recomendado"] in {"Escenario 1", "Escenario 2", "Escenario 3"}

def test_cambiar_distribucion_iteraciones_y_escenarios():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    for d in ("PERT", "Uniforme", "Normal", "Triangular"):
        at.selectbox(key="c_dist").set_value(d).run()
        assert not at.exception, (d, at.exception)
    at.select_slider(key="n_it").set_value(30000).run()
    [b for b in at.button if "Añadir escenario (copia" in b.label][0].click().run()
    assert len(at.session_state["esc"]) == 4
    [b for b in at.sidebar.button if "Simular" in b.label][0].click().run()
    assert not at.exception, at.exception
    assert len(at.session_state["sim"]["resultados"]) == 4
    assert len(at.session_state["sim"]["resultados"]["Escenario 1"]) == 30000

def test_error_de_validacion_bloquea():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    at.number_input(key="c_min").set_value(98.0).run()   # mínimo > máximo
    assert at.sidebar.button[0].disabled


def test_teoria_y_pdf_desde_la_app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    [b for b in at.sidebar.button if "Simular" in b.label][0].click().run()
    for nombre in ["Triangular", "PERT", "Uniforme", "Normal", "Lognormal", "Beta", "Discreta"]:
        at.selectbox(key="th_dist").set_value(nombre).run()
        assert not at.exception, (nombre, at.exception)
    at.button(key="gen_pdf").click().run()
    assert not at.exception, at.exception
    assert at.session_state["pdf"][1][:4] == b"%PDF"
