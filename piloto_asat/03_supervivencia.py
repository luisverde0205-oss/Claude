"""Paso 3. Curvas de supervivencia observadas (cuántos fragmentos siguen en órbita con el tiempo).

Entrada: datos_crudos/satcat.csv. Salida: resultados/supervivencia_resumen.csv y fragmentos_pruebas.csv.
"""
import numpy as np, pandas as pd
import comun as c

s = c.satcat()
hoy = pd.Timestamp('2026-09-30')
filas, todos = [], []
for p in c.PRUEBAS:
    g, previos, corte = c.fragmentos_de(s, p['intdes'], p['fecha'])
    t = pd.Timestamp(p['fecha'])
    g['vida_anios'] = (g.DECAY_DATE.fillna(hoy) - t).dt.days / 365.25
    g['en_orbita'] = g.DECAY_DATE.isna()
    g['evento'] = p['evento']
    todos.append(g)
    filas.append(dict(evento=p['evento'], satelite=p['satelite'], altitud_km=p['altitud_km'], corte_num_catalogo=corte,
                      fragmentos=len(g), excluidos_previos=previos, en_orbita_hoy=int(g.en_orbita.sum()),
                      pct_en_orbita_hoy=round(100 * g.en_orbita.mean(), 1),
                      vida_mediana_anios=round(float(np.median(g.vida_anios)), 3) if g.en_orbita.mean() < 0.5 else None,
                      pct_cayo_1_anio=round(100 * ((~g.en_orbita) & (g.vida_anios <= 1)).mean(), 1),
                      ultimo_reingreso=str(g.DECAY_DATE.max().date()) if (~g.en_orbita).any() else None))
r = pd.DataFrame(filas)
r.to_csv(c.RES / 'supervivencia_resumen.csv', index=False)
pd.concat(todos).to_csv(c.RES / 'fragmentos_pruebas.csv', index=False)
print(r.to_string(index=False))
