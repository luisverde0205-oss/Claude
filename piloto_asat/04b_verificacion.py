"""Paso 4b. Verificación del modelo contra datos que NO se usaron para calibrarlo.

1) China: fracción de fragmentos que cayó en la década pasada (observado, SATCAT) contra la
   fracción que el modelo predice para la próxima década, con y sin calibración.
2) Vida mediana observada de cada prueba (paso 3) contra la vida del modelo a esa altitud
   con el C_D A/m calibrado típico (órbita circular; aproximación gruesa).
"""
import json
import numpy as np, pandas as pd
import comun as c

f = pd.read_csv(c.RES / 'fragmentos_pruebas.csv', parse_dates=['DECAY_DATE'])
ch = f[f.evento == 'China 2007']
t0, t1 = pd.Timestamp('2016-10-01'), pd.Timestamp('2026-09-30')
vivos_t0 = ch[ch.DECAY_DATE.isna() | (ch.DECAY_DATE > t0)]
cayeron = vivos_t0.DECAY_DATE.between(t0, t1)
obs_decada = float(cayeron.mean())

def frac_cae_10(nombre):
    s = pd.read_csv(c.RES / f'proyeccion_fengyun1c_{nombre}.csv', parse_dates=['fecha'])
    n0 = s.en_orbita.iloc[0]; n10 = s[s.fecha <= '2036-10-01'].en_orbita.iloc[-1]
    return float(1 - n10 / n0)

res = {'china_decada': dict(observado_2016_2026=round(obs_decada, 3),
                            modelo_calibrado_2026_2036=round(frac_cae_10('observado'), 3),
                            modelo_sin_calibrar_2026_2036=round(frac_cae_10('observado_sin_calibrar'), 3),
                            nota='poblaciones distintas (los que quedan hoy son los más resistentes) y ciclos solares distintos: comparación de orden de magnitud')}

sup = pd.read_csv(c.RES / 'supervivencia_resumen.csv')
curva = pd.read_csv(c.RES / 'vida_contra_altitud.csv')
col50 = [k for k in curva.columns if k.startswith('vida_anios_p50')][0]
col16 = [k for k in curva.columns if k.startswith('vida_anios_p16')][0]
col84 = [k for k in curva.columns if k.startswith('vida_anios_p84')][0]
filas = []
for r in sup.itertuples():
    if pd.isna(r.vida_mediana_anios):
        continue
    interp = lambda col: float(np.exp(np.interp(r.altitud_km, curva.altitud_km, np.log(curva[col]))))
    filas.append(dict(evento=r.evento, altitud_km=r.altitud_km, vida_mediana_obs_dias=round(r.vida_mediana_anios * 365.25),
                      vida_modelo_dias_p50=round(interp(col50) * 365.25), rango_modelo_dias=[round(interp(col84) * 365.25), round(interp(col16) * 365.25)]))
res['vida_mediana_por_prueba'] = filas
(c.RES / 'verificacion_resumen.json').write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(res, indent=2, ensure_ascii=False))
