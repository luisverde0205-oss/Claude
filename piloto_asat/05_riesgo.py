"""Paso 5. Análisis de riesgo de colisión: método de densidad espacial ("gas cinético").

Método (Kessler y Cour-Palais, 1978):
 1. Para cada objeto en órbita, la fracción del periodo que pasa en cada capa de altura se
    obtiene con la ecuación de Kepler (segunda ley). Un objeto en órbita elíptica pasa más
    tiempo cerca del apogeo que del perigeo.
 2. Densidad espacial de la capa k:  S_k = sum_i (fracción del tiempo del objeto i en k) / V_k,
    con V_k = (4/3) pi (r_{k+1}^3 - r_k^3)  [objetos/km^3].
 3. Flujo sobre un satélite a la altura h:  F = S(h) · v_rel · A   [impactos/s],
    v_rel = 10 km/s (velocidad relativa media típica en órbita baja), A = área de choque.
 4. Número esperado de choques en un tiempo t: N = F·t; probabilidad P = 1 - exp(-N).
Supuestos (se reportan como límites): se ignora la distribución en latitud (inclinación),
solo cuenta objetos catalogados (mayores de ~10 cm) y v_rel es un valor típico, no calculado.

Entradas: datos_crudos/satcat.csv; resultados/estados_fengyun1c.npz (paso 4).
Salidas:  resultados/densidad_espacial.csv, riesgo_resumen.json, riesgo_fengyun1c_tiempo.csv.
"""
import json
import numpy as np, pandas as pd
import comun as c

V_REL = 10.0            # km/s
AREA_M2 = 10.0          # satélite típico
H = np.arange(200.0, 2001.0, 10.0)
R = c.RE + H
VOL = 4 / 3 * np.pi * (R[1:] ** 3 - R[:-1] ** 3)
CENTROS = (H[1:] + H[:-1]) / 2


def densidad_espacial(a, e):
    """Densidad espacial por capa [objetos/km^3] a partir de a [km] y e (arreglos)."""
    a = np.asarray(a, float)[:, None]; e = np.asarray(e, float)[:, None]
    F = c.fraccion_tiempo_bajo(R[None, :], a, e)
    circ = (e.ravel() < 1e-6)
    if circ.any():                                  # órbitas circulares: todo el tiempo en su capa
        F[circ] = (R[None, :] >= a[circ]).astype(float)
    return np.diff(F, axis=1).sum(axis=0) / VOL


def tasa_anual(S, area_m2=AREA_M2):
    return S * V_REL * (area_m2 * 1e-6) * c.ANIO


if __name__ == '__main__':
    s = c.satcat()
    orb = s[s.DECAY_DATE.isna() & (s.ORBIT_CENTER == 'EA') & s.PERIGEE.notna() & s.APOGEE.notna()
            & (s.PERIGEE > 100) & (s.APOGEE < 5000)].copy()
    rp, ra = orb.PERIGEE + c.RE, orb.APOGEE + c.RE
    orb['a'] = (rp + ra) / 2; orb['e'] = (ra - rp) / (ra + rp)
    orb['grupo'] = 'Resto de objetos'
    for p in c.PRUEBAS:
        g, _, _ = c.fragmentos_de(s, p['intdes'], p['fecha'])
        orb.loc[orb.NORAD_CAT_ID.isin(g.NORAD_CAT_ID), 'grupo'] = f"Prueba ASAT {p['evento']}"
    g, _, _ = c.fragmentos_de(s, c.COLISION_2009['intdes'], c.COLISION_2009['fecha'])
    orb.loc[orb.NORAD_CAT_ID.isin(g.NORAD_CAT_ID), 'grupo'] = 'Colisión accidental 2009'

    tabla = pd.DataFrame({'altitud_km': CENTROS})
    for grupo, gg in orb.groupby('grupo'):
        tabla[grupo] = densidad_espacial(gg.a, gg.e)
    grupos = [k for k in tabla.columns if k != 'altitud_km']
    tabla['Total'] = tabla[grupos].sum(axis=1)
    tabla.to_csv(c.RES / 'densidad_espacial.csv', index=False)

    res = {'objetos_en_orbita_usados': int(len(orb)), 'por_grupo': orb.grupo.value_counts().to_dict(),
           'supuestos': dict(v_rel_km_s=V_REL, area_m2=AREA_M2, capa_km=10)}
    obj = {}
    for h in (420, 550, 800, 850):
        fila = tabla.iloc[(tabla.altitud_km - h).abs().argmin()]
        tot = tasa_anual(fila.Total)
        obj[f'{h}_km'] = dict(choques_por_anio_10m2=float(tot), prob_5_anios=float(1 - np.exp(-5 * tot)),
                              años_entre_choques=float(1 / tot),
                              fraccion_por_grupo={k: round(float(fila[k] / fila.Total), 3) for k in grupos})
    res['riesgo_por_altitud'] = obj

    # Evolución del riesgo que dejan los fragmentos chinos (proyección del paso 4)
    est = np.load(c.RES / 'estados_fengyun1c.npz', allow_pickle=True)
    filas = []
    for fch, a, e, vivo in zip(est['fechas'], est['a'], est['e'], est['vivo']):
        S = densidad_espacial(a[vivo], e[vivo])
        i800 = np.abs(CENTROS - 800).argmin(); i550 = np.abs(CENTROS - 550).argmin()
        filas.append(dict(fecha=fch, fragmentos=int(vivo.sum()),
                          choques_anio_10m2_800km=float(tasa_anual(S[i800])),
                          choques_anio_10m2_550km=float(tasa_anual(S[i550]))))
    t = pd.DataFrame(filas); t.to_csv(c.RES / 'riesgo_fengyun1c_tiempo.csv', index=False)
    anios = pd.to_datetime(t.fecha).dt.year
    res['fengyun1c_futuro'] = dict(
        nota='solo los 1,932 fragmentos con elementos y dn/dt válidos; escalar x1.2 para los 2,315 en órbita',
        choques_esperados_10m2_800km_2026_2126=float(t.choques_anio_10m2_800km.sum()),
        tasa_800km_2026=float(t.choques_anio_10m2_800km.iloc[0]), tasa_800km_2050=float(t.choques_anio_10m2_800km[anios == 2050].iloc[0]),
        tasa_800km_2100=float(t.choques_anio_10m2_800km[anios == 2100].iloc[0]))
    (c.RES / 'riesgo_resumen.json').write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, indent=2, ensure_ascii=False))
