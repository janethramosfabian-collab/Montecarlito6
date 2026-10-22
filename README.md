# Plataforma Montecarlo para planeamiento minero (v3)

Compara escenarios de producción con incertidumbre en la **certeza geo-operativa** y calcula
**qué escenario tiene más probabilidad de cumplir el plan**.

## Regla de la v3
- Los datos son los 3 escenarios del Excel «Escenarios para Montecarlo» (TMS, ley base, recuperación, precio, costo y certeza por zona).
- **Ley estimada = ley base + (ley base − ley base / certeza) = ley base × (2 − 1/certeza)**.
- El determinístico (plan) usa la certeza de la tabla (70 %, 88 % y 97 % por zona).
- **Lo único que se sortea en cada iteración es la certeza geo-operativa** de cada zona, entre 70 % y 97 % (editable).
  Precio, TMS, ley base, recuperación y costos son fijos.
- Iteraciones: 20 000 · 30 000 · 50 000 · 100 000.
- Éxito = probabilidad de que la utilidad alcance al menos X % del plan (100 % por defecto; se elige en la barra lateral).

## Pestañas
1 Escenarios (tablas con el formato del Excel) · 2 Incertidumbre (distribución de la certeza, gráfico certeza → ley) ·
3 Resultados por escenario · 4 Comparación lado a lado (incluye certeza plan vs. simulada) · 5 Teoría ·
6 Informe PDF con firma y exportación (Excel / CSV).

## Ejecutar
```
pip install -r requirements.txt
streamlit run app.py
pytest tests
```
