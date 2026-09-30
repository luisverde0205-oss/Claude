"""Funciones compartidas: constantes, lectura de datos crudos, leyes de Kepler,
densidad atmosférica y frenado promediado sobre la órbita."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

RAIZ = Path(__file__).resolve().parent
CRUDOS = RAIZ / 'datos_crudos'
RES = RAIZ / 'resultados'
FIG = RAIZ / 'figuras'

MU = 398600.4418          # km^3/s^2, parámetro gravitacional de la Tierra (G·M)
RE = 6378.137             # km, radio ecuatorial (WGS-84); el catálogo reporta alturas sobre este radio
DIA = 86400.0             # s
ANIO = 365.25 * DIA       # s
BSTAR_A_BC = 12.741621    # (C_D·A/m en m^2/kg) = 12.741621 · B* (B* en 1/radio terrestre); Vallado (2013)

# Pruebas antisatélite: designador internacional del satélite destruido y fecha de la prueba.
PRUEBAS = [
    dict(evento='China 2007', satelite='Fengyun-1C', intdes='1999-025', fecha='2007-01-11', altitud_km=865),
    dict(evento='EE.UU. 2008', satelite='USA-193', intdes='2006-057', fecha='2008-02-21', altitud_km=247),
    dict(evento='India 2019', satelite='Microsat-R', intdes='2019-006', fecha='2019-03-27', altitud_km=283),
    dict(evento='Rusia 2021', satelite='Kosmos 1408', intdes='1982-092', fecha='2021-11-15', altitud_km=480),
]
# Colisión accidental de 2009 (Iridium 33 contra Kosmos 2251), para comparar.
COLISION_2009 = dict(evento='Colisión 2009', intdes=('1993-036', '1997-051'), fecha='2009-02-10')


# ---------------------------------------------------------------- datos
def satcat():
    s = pd.read_csv(CRUDOS / 'satcat.csv')
    s['LAUNCH_DATE'] = pd.to_datetime(s.LAUNCH_DATE, errors='coerce')
    s['DECAY_DATE'] = pd.to_datetime(s.DECAY_DATE, errors='coerce')
    return s


def fragmentos_de(s, intdes, fecha):
    """Fragmentos (tipo DEB) catalogados después de la prueba.

    El número de catálogo crece con el tiempo. Como corte se usa la mediana del número
    asignado a objetos no-DEB lanzados en los 30 días previos a la prueba; los fragmentos
    con número menor ya existían antes de la prueba y se excluyen.
    """
    t = pd.Timestamp(fecha)
    previos = s[(s.LAUNCH_DATE >= t - pd.Timedelta(days=30)) & (s.LAUNCH_DATE < t) & (s.OBJECT_TYPE != 'DEB')]
    corte = previos.NORAD_CAT_ID.median()
    intdes = (intdes,) if isinstance(intdes, str) else intdes
    g = s[s.OBJECT_ID.str.startswith(intdes) & (s.OBJECT_TYPE == 'DEB')]
    return g[g.NORAD_CAT_ID > corte].copy(), int((g.NORAD_CAT_ID <= corte).sum()), float(corte)


def clima_espacial():
    """F10.7 (flujo solar a 10.7 cm) y Ap (índice geomagnético) diarios, observados y predichos."""
    w = pd.read_csv(CRUDOS / 'SW-All.csv', parse_dates=['DATE'])
    w = w.set_index('DATE')
    f107 = w['F10.7_OBS'].astype(float)
    f81 = w['F10.7_OBS_CENTER81'].astype(float).fillna(f107)
    ap = w['AP_AVG'].astype(float)
    tipo = w['F10.7_DATA_TYPE']
    return pd.DataFrame(dict(f107=f107, f107_81=f81, ap=ap, tipo=tipo))


# ---------------------------------------------------------------- Kepler
def a_desde_n(n_rev_dia):
    """Tercera ley de Kepler: n^2 a^3 = mu  ->  a = (mu / n^2)^(1/3)   [km]."""
    n = np.asarray(n_rev_dia, float) * 2 * np.pi / DIA
    return (MU / n ** 2) ** (1 / 3)


def periodo_min(a_km):
    """Tercera ley de Kepler: T = 2 pi sqrt(a^3 / mu)   [min]."""
    return 2 * np.pi * np.sqrt(np.asarray(a_km, float) ** 3 / MU) / 60


def perigeo_apogeo(a_km, e):
    """Primera ley (órbita elíptica): r_p = a(1-e), r_a = a(1+e); se devuelven alturas [km]."""
    return a_km * (1 - e) - RE, a_km * (1 + e) - RE


def vis_viva(r_km, a_km):
    """Conservación de la energía: v^2 = mu (2/r - 1/a)   [km/s]."""
    return np.sqrt(MU * (2 / r_km - 1 / a_km))


def fraccion_tiempo_bajo(r_km, a_km, e):
    """Fracción del periodo que el objeto pasa a radio <= r (segunda ley vía ecuación de Kepler).

    r = a(1 - e cos E) define la anomalía excéntrica E; la ecuación de Kepler M = E - e sin E
    da la anomalía media, proporcional al tiempo (áreas iguales en tiempos iguales).
    Entre perigeo y apogeo la fracción es M/pi.
    """
    r_km, a_km, e = np.broadcast_arrays(np.asarray(r_km, float), np.asarray(a_km, float), np.asarray(e, float))
    rp, ra = a_km * (1 - e), a_km * (1 + e)
    out = np.where(r_km >= ra, 1.0, 0.0)
    dentro = (r_km > rp) & (r_km < ra) & (e > 0)
    cosE = np.clip((1 - r_km[dentro] / a_km[dentro]) / e[dentro], -1, 1)
    E = np.arccos(cosE)
    out[dentro] = (E - e[dentro] * np.sin(E)) / np.pi
    return out


# ---------------------------------------------------------------- atmósfera
ALT_TABLA = np.arange(100.0, 2501.0, 10.0)
F107_TABLA = np.arange(60.0, 301.0, 20.0)
AP_TABLA = np.array([4.0, 15.0, 50.0])


def tabla_densidad(recalcular=False):
    """log10 de la densidad media global (kg/m^3) de NRLMSISE-00 en una malla (altura, F10.7, Ap).

    Se promedia sobre una rejilla de latitudes y longitudes (peso cos lat) en un equinoccio,
    lo que aproxima el promedio a lo largo de muchas órbitas. Los índices solares se pasan
    explícitamente, así que no se descarga nada adicional.
    """
    ruta = RES / 'tabla_densidad_msis.npz'
    if ruta.exists() and not recalcular:
        return np.load(ruta)['logrho']
    import pymsis
    lats = np.array([-75, -45, -15, 15, 45, 75.0]); lons = np.arange(0, 360, 45.0)
    LA, LO = np.meshgrid(lats, lons); w = np.cos(np.radians(LA)).ravel(); npt = LA.size
    logrho = np.zeros((ALT_TABLA.size, F107_TABLA.size, AP_TABLA.size))
    fecha = np.array(['2020-03-20T12:00'] * npt, dtype='datetime64[s]')
    for j, f in enumerate(F107_TABLA):
        for k, ap in enumerate(AP_TABLA):
            m = npt * ALT_TABLA.size
            H = np.repeat(ALT_TABLA, npt)
            out = pymsis.calculate(np.tile(fecha, ALT_TABLA.size), np.tile(LO.ravel(), ALT_TABLA.size),
                                   np.tile(LA.ravel(), ALT_TABLA.size), H,
                                   f107s=np.full(m, f), f107as=np.full(m, f), aps=np.full((m, 7), ap))
            r = out[..., pymsis.Variable.MASS_DENSITY].reshape(ALT_TABLA.size, npt)
            logrho[:, j, k] = np.log10((r * w).sum(axis=1) / w.sum())
    RES.mkdir(exist_ok=True)
    np.savez(ruta, logrho=logrho, alt=ALT_TABLA, f107=F107_TABLA, ap=AP_TABLA)
    return logrho


def densidad(h_km, f107, ap, logrho):
    """Interpolación trilineal en log10(rho); devuelve kg/m^3. h_km puede ser un arreglo."""
    from scipy.interpolate import RegularGridInterpolator
    it = RegularGridInterpolator((ALT_TABLA, F107_TABLA, AP_TABLA), logrho, bounds_error=False, fill_value=None)
    h = np.clip(np.asarray(h_km, float), ALT_TABLA[0], ALT_TABLA[-1])
    pts = np.stack([h, np.full(h.shape, np.clip(f107, 60, 300)), np.full(h.shape, np.clip(ap, 4, 50))], axis=-1)
    return 10 ** it(pts.reshape(-1, 3)).reshape(h.shape)


def perfil_log(f107, ap, logrho):
    """Perfil log10(rho) contra ALT_TABLA para un F10.7 y Ap dados (interpolación bilineal)."""
    f, a = float(np.clip(f107, F107_TABLA[0], F107_TABLA[-1])), float(np.clip(ap, AP_TABLA[0], AP_TABLA[-1]))
    j = min(np.searchsorted(F107_TABLA, f, side='right') - 1, F107_TABLA.size - 2)
    k = min(np.searchsorted(AP_TABLA, a, side='right') - 1, AP_TABLA.size - 2)
    tf = (f - F107_TABLA[j]) / (F107_TABLA[j + 1] - F107_TABLA[j])
    ta = (a - AP_TABLA[k]) / (AP_TABLA[k + 1] - AP_TABLA[k])
    return ((1 - tf) * (1 - ta) * logrho[:, j, k] + tf * (1 - ta) * logrho[:, j + 1, k]
            + (1 - tf) * ta * logrho[:, j, k + 1] + tf * ta * logrho[:, j + 1, k + 1])


def indices_solares(fechas, clima, escenario='observado'):
    """F10.7 (media de 81 días) y Ap para cada fecha.

    'observado': datos observados; después, las predicciones mensuales del archivo (Ap = 15);
    más allá de la última predicción (2041) se repite la serie en ciclos de 11 años.
    Un número en lugar de 'observado' fija F10.7 constante (Ap = 15) para escenarios de sensibilidad.
    """
    fechas = pd.to_datetime(pd.Series(fechas))
    if escenario != 'observado':
        return np.full(len(fechas), float(escenario)), np.full(len(fechas), 15.0)
    serie, ultima_obs = _serie_diaria(clima)
    ultima = serie.index.max()
    f = fechas.copy()
    while (f > ultima).any():
        f = f.where(f <= ultima, f - pd.DateOffset(years=11))
    f107 = serie.f107_81.reindex(f.values).to_numpy()
    ap = serie.ap.reindex(f.values).to_numpy()
    ap = np.where((f.values > np.datetime64(ultima_obs)) | np.isnan(ap), 15.0, ap)
    return f107, ap


_SERIE = {}
def _serie_diaria(clima):
    """Serie diaria continua: las predicciones mensuales se interpolan linealmente entre meses."""
    if 'x' not in _SERIE:
        num = clima[['f107', 'f107_81', 'ap']]
        serie = num.reindex(pd.date_range(num.index.min(), num.index.max(), freq='D')).interpolate(limit_direction='both')
        _SERIE['x'] = (serie, clima.index[clima.tipo == 'OBS'].max())
    return _SERIE['x']


# ---------------------------------------------------------------- frenado (ecuaciones de Gauss promediadas)
NQ = 48
_E = (np.arange(NQ) + 0.5) * 2 * np.pi / NQ   # anomalía excéntrica en puntos equiespaciados


def tasas_frenado(a_km, e, bc_m2_kg, f107, ap, logrho):
    """Promedio orbital de da/dt [km/s] y de/dt [1/s] por frenado atmosférico tangencial.

    Energía: dE/dt = F·v = -(1/2) rho B v^3,  E = -mu/(2a)  ->  da/dt = -(a^2/mu) rho B v^3
    Excentricidad (Gauss, fuerza tangencial): de/dt = -rho B v (e + cos nu)
    B = C_D A/m. El promedio en el tiempo usa dt = (1 - e cos E)/n dE (ecuación de Kepler).
    Se ignora la rotación de la atmósfera (efecto de unos pocos por ciento).
    """
    a = np.asarray(a_km, float)[:, None]; ee = np.asarray(e, float)[:, None]
    B = np.asarray(bc_m2_kg, float)[:, None] * 1e-6          # m^2/kg -> km^2/kg
    cosE = np.cos(_E)[None, :]
    r = a * (1 - ee * cosE)
    v = np.sqrt(MU * (2 / r - 1 / a))                        # km/s
    rho = 10 ** np.interp(r - RE, ALT_TABLA, perfil_log(f107, ap, logrho)) * 1e9   # kg/m^3 -> kg/km^3
    peso = (1 - ee * cosE) / NQ                               # dt/T para cada punto
    cosnu = (cosE - ee) / (1 - ee * cosE)
    dadt = -(a ** 2 / MU) * np.sum(peso * rho * B * v ** 3, axis=1)[:, None]
    dedt = -np.sum(peso * rho * B * v * (ee + cosnu), axis=1)[:, None]
    return dadt.ravel(), dedt.ravel()
