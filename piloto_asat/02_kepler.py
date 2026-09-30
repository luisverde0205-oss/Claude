"""Paso 2. Leyes de Kepler aplicadas a los fragmentos chinos que siguen en órbita.

Entrada: datos_crudos/gp_fengyun1c_1999-025.csv (movimiento medio n, excentricidad e, B*).
Salida:  resultados/kepler_fengyun1c.csv y resultados/kepler_resumen.json.
Control: las alturas y periodos calculados se comparan con los que publica el catálogo SATCAT.
"""
import json
import numpy as np, pandas as pd
import comun as c

gp = pd.read_csv(c.CRUDOS / 'gp_fengyun1c_1999-025.csv')
s = c.satcat()
frag, _, corte = c.fragmentos_de(s, '1999-025', '2007-01-11')
gp = gp[gp.NORAD_CAT_ID.isin(frag.NORAD_CAT_ID)].copy()

gp['a_km'] = c.a_desde_n(gp.MEAN_MOTION)                         # 3a ley
gp['T_min'] = c.periodo_min(gp.a_km)                             # 3a ley
gp['h_perigeo_km'], gp['h_apogeo_km'] = c.perigeo_apogeo(gp.a_km, gp.ECCENTRICITY)   # 1a ley
gp['v_perigeo_kms'] = c.vis_viva(gp.a_km * (1 - gp.ECCENTRICITY), gp.a_km)
gp['v_apogeo_kms'] = c.vis_viva(gp.a_km * (1 + gp.ECCENTRICITY), gp.a_km)
# 2a ley: fracción del tiempo que pasa por debajo de su altura media
gp['frac_tiempo_bajo_a'] = c.fraccion_tiempo_bajo(gp.a_km, gp.a_km, gp.ECCENTRICITY)
gp['CdA_m_m2kg'] = c.BSTAR_A_BC * gp.BSTAR

m = gp.merge(s[['NORAD_CAT_ID', 'PERIGEE', 'APOGEE', 'PERIOD']], on='NORAD_CAT_ID')
dp, da, dT = m.h_perigeo_km - m.PERIGEE, m.h_apogeo_km - m.APOGEE, m.T_min - m.PERIOD
res = dict(
    fragmentos_con_elementos=int(len(gp)),
    comparados_con_satcat=int(len(m)),
    dif_perigeo_km_mediana=float(np.median(dp)), dif_perigeo_km_p95_abs=float(np.percentile(abs(dp), 95)),
    dif_apogeo_km_mediana=float(np.median(da)), dif_apogeo_km_p95_abs=float(np.percentile(abs(da), 95)),
    dif_periodo_min_mediana=float(np.median(dT)), dif_periodo_min_p95_abs=float(np.percentile(abs(dT), 95)),
    perigeo_km_cuantiles_5_50_95=np.percentile(gp.h_perigeo_km, [5, 50, 95]).round(1).tolist(),
    apogeo_km_cuantiles_5_50_95=np.percentile(gp.h_apogeo_km, [5, 50, 95]).round(1).tolist(),
    excentricidad_mediana=float(gp.ECCENTRICITY.median()),
    v_perigeo_menos_v_apogeo_kms_mediana=float((gp.v_perigeo_kms - gp.v_apogeo_kms).median()),
    frac_tiempo_bajo_a_mediana=float(gp.frac_tiempo_bajo_a.median()),
    fraccion_bstar_positivo=float((gp.BSTAR > 0).mean()),
    CdA_m_m2kg_cuantiles_5_50_95_bstar_pos=np.percentile(gp.CdA_m_m2kg[gp.BSTAR > 0], [5, 50, 95]).round(4).tolist(),
    epoca_min=str(gp.EPOCH.min()), epoca_max=str(gp.EPOCH.max()),
)
c.RES.mkdir(exist_ok=True)
gp.to_csv(c.RES / 'kepler_fengyun1c.csv', index=False)
(c.RES / 'kepler_resumen.json').write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(res, indent=2, ensure_ascii=False))
