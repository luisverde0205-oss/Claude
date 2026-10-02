# Mecánica estadística del dinero: simulación

Reproducción sencilla del modelo de A. Drăgulescu y V. M. Yakovenko,
*"Statistical mechanics of money"*, Eur. Phys. J. B 17, 723–729 (2000).

## Qué hace
- N personas empiezan con el mismo dinero.
- En cada paso se escoge al azar quién paga y quién recibe; el que paga entrega una cantidad fija.
- Si el que paga no tiene dinero suficiente, el intercambio no ocurre (no hay deudas).
- El dinero total se conserva, igual que la energía en un gas aislado.

Predicción: el dinero termina repartido según la ley de Boltzmann, P(m) ∝ exp(−m/T),
con T = dinero promedio por persona. El Gini teórico es 0.5.

## Cómo correrlo
```bash
pip install -r requirements.txt
python simulacion_dinero.py
```
Genera `figuras/simulacion_dinero.png` e imprime la comparación con la teoría.

## Datos reales (siguiente paso)
- **ENIGH (INEGI)**: ingreso corriente de los hogares de México (tabla `concentradohogar`, usar el factor de expansión).
- **World Inequality Database (wid.world)**: corrige la parte de los más ricos, que las encuestas subestiman.

## Documento modular
- `modelo.py`: núcleo del modelo (simulación, entropía, Gini, Lorenz, conteo exacto).
- `generar_resultados.py`: genera `figuras/fig1..fig5` y `resultados/*.csv` (≈1 min).
- `deudas.py`: modelo con deuda (límite m_d); genera `fig8_deudas.png` y `resultados/deudas.csv` (≈2.5 min).
- `comparar_paises.py`: descarga Gini y participaciones del Banco Mundial (datos reales) y genera `fig6`, `fig7`.
- `modular/modular_dinero.tex` y `.pdf`: documento en la plantilla de proyecto modular (Lic. Física, CUCEI).
  Compilar desde `modular/` con `pdflatex modular_dinero.tex` (dos veces).

Los resultados del modelo son **datos simulados**; solo `comparar_paises.py` usa datos reales.
