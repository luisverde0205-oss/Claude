"""Pruebas numéricas: cada función física contra un resultado conocido a mano."""
import numpy as np
import comun as c


def test_tercera_ley():
    # Una órbita de 15.5 vueltas por día (típica a ~420 km, como la Estación Espacial)
    a = c.a_desde_n(15.5)
    assert abs((a - c.RE) - 420) < 15, a
    # Ida y vuelta: T(a) debe devolver 1440/15.5 minutos
    assert abs(c.periodo_min(a) - 1440 / 15.5) < 1e-9


def test_perigeo_apogeo():
    hp, ha = c.perigeo_apogeo(7000.0, 0.01)
    assert abs(hp - (6930 - c.RE)) < 1e-9 and abs(ha - (7070 - c.RE)) < 1e-9


def test_vis_viva_circular():
    # En órbita circular v = sqrt(mu/r)
    assert abs(c.vis_viva(7000.0, 7000.0) - np.sqrt(c.MU / 7000.0)) < 1e-12


def test_segunda_ley():
    a, e = 7200.0, 0.05
    rp, ra = a * (1 - e), a * (1 + e)
    assert c.fraccion_tiempo_bajo(rp, a, e) == 0.0 and c.fraccion_tiempo_bajo(ra, a, e) == 1.0
    # Por la segunda ley, pasa MENOS de la mitad del tiempo por debajo del radio medio (a):
    f = float(c.fraccion_tiempo_bajo(a, a, e))
    assert abs(f - (np.pi / 2 - e) / np.pi) < 1e-12 and f < 0.5
    # La fracción, estimada con una simulación de tiempo uniforme, coincide con la ecuación de Kepler
    M = np.linspace(0, 2 * np.pi, 200001)[:-1]
    E = M.copy()
    for _ in range(50):
        E = M + e * np.sin(E)
    r = a * (1 - e * np.cos(E))
    assert abs(np.mean(r <= a) - f) < 1e-4


def test_conversion_bstar():
    # B* = 1e-4 (1/radio terrestre) -> C_D A/m = 1.274e-3 m^2/kg
    assert abs(c.BSTAR_A_BC * 1e-4 - 1.2741621e-3) < 1e-12


def test_frenado_circular():
    # En órbita circular, el promedio de Gauss debe reducirse a da/dt = -rho B sqrt(mu a) y de/dt = 0
    L = c.tabla_densidad()
    a, B = c.RE + 500, 0.02
    dadt, dedt = c.tasas_frenado([a], [0.0], [B], 150, 15, L)
    rho = c.densidad(np.array([500.0]), 150, 15, L)[0] * 1e9
    esperado = -rho * B * 1e-6 * np.sqrt(c.MU * a)
    assert abs(dadt[0] / esperado - 1) < 1e-6 and abs(dedt[0]) < 1e-20


def test_densidad_monotona():
    L = c.tabla_densidad()
    r = c.densidad(np.array([200, 400, 800, 1600.0]), 150, 15, L)
    assert np.all(np.diff(r) < 0)
    # Más actividad solar -> atmósfera más densa a 500 km
    assert c.densidad(np.array([500.0]), 250, 15, L)[0] > c.densidad(np.array([500.0]), 70, 15, L)[0]


if __name__ == '__main__':
    for nombre, f in list(globals().items()):
        if nombre.startswith('test_'):
            f(); print('PASS', nombre)
