"""Paso 4. Modelo de decaimiento orbital por frenado atmosférico.

4a. Control y calibración: la tasa de cambio del movimiento medio que predice el modelo
    (dn/dt) con C_D A/m = 12.74 B* se compara con la observada que publica el catálogo
    (MEAN_MOTION_DOT = (dn/dt)/2). Por la tercera ley, n = sqrt(mu/a^3) -> dn/dt = -(3/2)(n/a) da/dt.
    B* no es un coeficiente físico: el catálogo lo ajusta con una atmósfera fija (la de SGP4),
    demasiado densa a gran altura, así que la razón modelo/observado cae con la altitud.
    Por eso el C_D A/m físico de cada fragmento se CALIBRA con su dn/dt observado:
    C_D A/m = 12.74 B* x (dn/dt observado) / (dn/dt modelo).
4b. Proyección: se integran a y e de cada fragmento chino hacia el futuro (paso de 5 días)
    con la actividad solar observada/predicha y dos escenarios constantes de sensibilidad.
4c. Vida orbital contra altitud inicial (órbitas circulares) para la discusión de normas.
4d. Predicción verificable: fecha de reingreso de los 4 fragmentos rusos que quedan.
"""
import json
import numpy as np, pandas as pd
import comun as c

L = c.tabla_densidad()
clima = c.clima_espacial()
H_FIN = 180.0          # km: con perigeo por debajo, el reingreso ocurre en días
PASO = 5 * c.DIA


def integrar(a0, e0, bc, inicio, anios, escenario='observado', guardar_cada=365, paso=PASO):
    """Devuelve (fechas de reingreso, serie anual del número de objetos en órbita)."""
    a, e = np.array(a0, float), np.array(e0, float)
    bc = np.array(bc, float)
    vivo = np.ones(a.size, bool); reingreso = np.full(a.size, np.datetime64('NaT'), 'datetime64[D]')
    pasos = int(anios * c.ANIO / paso)
    fechas = pd.date_range(inicio, periods=pasos + 1, freq=f'{int(paso / c.DIA)}D')
    f107, ap = c.indices_solares(fechas, clima, escenario)
    serie, estado = [], []
    for k in range(pasos):
        if k % max(1, int(guardar_cada * c.DIA / paso)) == 0:
            serie.append((fechas[k], int(vivo.sum())))
            estado.append((fechas[k], a.copy(), e.copy(), vivo.copy()))
        if not vivo.any():
            break
        i = np.flatnonzero(vivo)
        dadt, dedt = c.tasas_frenado(a[i], e[i], bc[i], f107[k], ap[k], L)
        a[i] += dadt * paso
        e[i] = np.clip(e[i] + dedt * paso, 0.0, 0.9)
        cae = a[i] * (1 - e[i]) - c.RE < H_FIN
        reingreso[i[cae]] = np.datetime64(fechas[k + 1].date(), 'D')
        vivo[i[cae]] = False
    return reingreso, pd.DataFrame(serie, columns=['fecha', 'en_orbita']), estado


if __name__ == '__main__':
    gp = pd.read_csv(c.RES / 'kepler_fengyun1c.csv')
    gp = gp[gp.BSTAR > 0].copy()           # B* <= 0 no tiene sentido físico para frenado
    res = {'fragmentos_modelados': int(len(gp))}

    # 4a. Control con dn/dt observado
    f107, ap = c.indices_solares(pd.to_datetime(gp.EPOCH).dt.normalize(), clima)
    dadt = np.array([c.tasas_frenado([a], [e], [b], f, p, L)[0][0]
                     for a, e, b, f, p in zip(gp.a_km, gp.ECCENTRICITY, gp.CdA_m_m2kg, f107, ap)])
    n = gp.MEAN_MOTION * 2 * np.pi / c.DIA
    ndot_mod = -1.5 * n / gp.a_km * dadt * (c.DIA ** 2 / (2 * np.pi))    # rev/día^2
    ndot_obs = 2 * gp.MEAN_MOTION_DOT
    ok = ndot_obs > 0
    razon = (ndot_mod[ok] / ndot_obs[ok])
    gp['ndot_modelo'] = ndot_mod; gp['ndot_observado'] = ndot_obs
    gp['razon_modelo_obs'] = ndot_mod / ndot_obs
    res['control_ndot'] = dict(n_comparados=int(ok.sum()), razon_modelo_observado_mediana=float(np.median(razon)),
                               razon_p16_p84=np.percentile(razon, [16, 84]).round(3).tolist(),
                               correlacion_log=float(np.corrcoef(np.log10(ndot_mod[ok]), np.log10(ndot_obs[ok]))[0, 1]),
                               F107_81_en_epoca=float(np.median(f107)))
    bins = pd.cut(gp.h_perigeo_km[ok], [500, 600, 700, 800, 900])
    res['control_ndot']['razon_mediana_por_perigeo'] = {str(k): round(float(v), 4) for k, v in razon.groupby(bins, observed=True).median().items()}
    gp = gp[ok].copy()
    gp['CdA_m_calibrado'] = gp.CdA_m_m2kg / gp.razon_modelo_obs
    res['CdA_m_calibrado_p16_p50_p84'] = np.percentile(gp.CdA_m_calibrado, [16, 50, 84]).round(4).tolist()
    print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)

    # 4b. Proyección de los fragmentos chinos
    inicio = '2026-10-01'
    proy = {}
    escenarios = [('observado', 'CdA_m_calibrado', 'actividad solar observada/predicha'),
                  (70.0, 'CdA_m_calibrado', 'Sol tranquilo (F10.7=70)'),
                  (200.0, 'CdA_m_calibrado', 'Sol activo (F10.7=200)'),
                  ('observado', 'CdA_m_m2kg', 'sin calibrar (C_D A/m = 12.74 B*)')]
    for esc, col, nombre in escenarios:
        rein, serie, estado = integrar(gp.a_km, gp.ECCENTRICITY, gp[col], inicio, 100, esc)
        etiqueta = str(esc).replace('.0', '') + ('' if col == 'CdA_m_calibrado' else '_sin_calibrar')
        serie.to_csv(c.RES / f'proyeccion_fengyun1c_{etiqueta}.csv', index=False)
        anios = (pd.to_datetime(rein) - pd.Timestamp(inicio)).days / 365.25
        proy[nombre] = {f'en_orbita_{y}': int((serie.en_orbita[serie.fecha.dt.year <= y]).iloc[-1]) for y in (2036, 2050, 2076, 2100, 2126)}
        proy[nombre]['vida_restante_mediana_anios'] = float(np.nanmedian(np.where(np.isnan(anios), np.inf, anios)))
        if esc == 'observado' and col == 'CdA_m_calibrado':
            gp['reingreso_modelo'] = rein
            # Estados anuales para el análisis de riesgo (paso 5)
            np.savez(c.RES / 'estados_fengyun1c.npz', fechas=np.array([str(f.date()) for f, *_ in estado]),
                     a=np.array([x for _, x, _, _ in estado]), e=np.array([x for _, _, x, _ in estado]),
                     vivo=np.array([x for _, _, _, x in estado]))
        print(nombre, proy[nombre], flush=True)
    res['proyeccion_fengyun1c'] = proy
    gp.to_csv(c.RES / 'decaimiento_fengyun1c.csv', index=False)

    # 4c. Vida contra altitud (órbita circular), con el C_D A/m típico de los fragmentos
    bcs = np.percentile(gp.CdA_m_calibrado, [16, 50, 84])
    alts = np.arange(250, 901, 25.0)
    curva = {'altitud_km': alts.tolist()}
    for etiqueta, b in zip(['p16', 'p50', 'p84'], bcs):
        rein, _, _ = integrar(c.RE + alts, np.zeros_like(alts), np.full(alts.size, b), inicio, 200, paso=c.DIA)
        v = (pd.to_datetime(rein) - pd.Timestamp(inicio)).days / 365.25
        curva[f'vida_anios_{etiqueta}_CdAm_{b:.4f}'] = np.where(np.isnan(v), np.inf, v).tolist()
    pd.DataFrame(curva).to_csv(c.RES / 'vida_contra_altitud.csv', index=False)
    res['vida_contra_altitud_CdAm_p16_p50_p84'] = bcs.round(4).tolist()
    print(pd.DataFrame(curva).round(2).to_string(index=False), flush=True)

    # 4d. Predicción verificable para los fragmentos rusos restantes
    ru = pd.read_csv(c.CRUDOS / 'gp_kosmos1408_1982-092.csv')
    ru = ru[ru.OBJECT_NAME.str.contains('DEB')].copy()
    ru['a_km'] = c.a_desde_n(ru.MEAN_MOTION); ru['CdA_m_m2kg'] = c.BSTAR_A_BC * ru.BSTAR
    fr, pr = c.indices_solares(pd.to_datetime(ru.EPOCH).dt.normalize(), clima)
    for idx, (r, f0, p0) in enumerate(zip(ru.itertuples(), fr, pr)):
        da0 = c.tasas_frenado([r.a_km], [r.ECCENTRICITY], [r.CdA_m_m2kg], f0, p0, L)[0][0]
        n0 = r.MEAN_MOTION * 2 * np.pi / c.DIA
        nd0 = -1.5 * n0 / r.a_km * da0 * (c.DIA ** 2 / (2 * np.pi))
        ru.loc[r.Index, 'CdA_m_m2kg'] = r.CdA_m_m2kg * (2 * r.MEAN_MOTION_DOT) / nd0
    ru['h_perigeo_km'], ru['h_apogeo_km'] = c.perigeo_apogeo(ru.a_km, ru.ECCENTRICITY)
    PASO_R = c.DIA
    pred = []
    for r in ru.itertuples():
        a, e, t = r.a_km, r.ECCENTRICITY, pd.Timestamp(r.EPOCH).normalize()
        for d in range(0, 3650):
            f, p = c.indices_solares([t + pd.Timedelta(days=d)], clima)
            da, de = c.tasas_frenado([a], [e], [r.CdA_m_m2kg], f[0], p[0], L)
            a += da[0] * PASO_R; e = max(0.0, e + de[0] * PASO_R)
            if a * (1 - e) - c.RE < 150:
                break
        pred.append(dict(NORAD=int(r.NORAD_CAT_ID), perigeo_km=round(r.h_perigeo_km, 1), apogeo_km=round(r.h_apogeo_km, 1),
                         CdA_m=round(r.CdA_m_m2kg, 5), epoca=str(t.date()), reingreso_predicho=str((t + pd.Timedelta(days=d)).date())))
    res['prediccion_fragmentos_rusos'] = pred
    print(pd.DataFrame(pred).to_string(index=False))
    (c.RES / 'decaimiento_resumen.json').write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
