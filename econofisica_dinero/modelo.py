"""
Núcleo del modelo de Drăgulescu y Yakovenko (2000) y funciones de medición.

Todos los datos que produce este módulo son SIMULADOS: no provienen de ninguna
encuesta ni registro real.
"""
from math import comb

import numpy as np


def simular(N, m0, dm=1, pasos=4_000_000, cada=20_000, semilla=0,
            regla="fijo", seguir=()):
    """Simula N personas que intercambian dinero al azar.

    regla="fijo":   el pagador entrega `dm` monedas; si no las tiene, no hay intercambio.
    regla="reparto": la pareja junta su dinero y lo reparte con una fracción al azar
                     (variante continua, también conserva el dinero).
    seguir:         índices de personas cuyo dinero se registra a lo largo del tiempo.

    Devuelve un diccionario con el dinero final y las series de tiempo medidas.
    """
    rng = np.random.default_rng(semilla)
    entero = regla == "fijo"
    dinero = np.full(N, m0, dtype=np.int64 if entero else float)
    total = dinero.sum()
    a = rng.integers(0, N, pasos)
    b = rng.integers(0, N, pasos)
    eps = rng.random(pasos) if regla == "reparto" else None

    tiempos, entropias, ginis = [], [], []
    trayectorias = {k: [] for k in seguir}
    for t in range(pasos):
        i, j = a[t], b[t]
        if i != j:
            if entero:
                if dinero[i] >= dm:
                    dinero[i] -= dm
                    dinero[j] += dm
            else:
                s = dinero[i] + dinero[j]
                dinero[i] = eps[t] * s
                dinero[j] = s - dinero[i]
        if t % cada == 0:
            tiempos.append(t / N)
            entropias.append(entropia(dinero, m0))
            ginis.append(gini(dinero))
            for k in seguir:
                trayectorias[k].append(dinero[k])

    assert np.isclose(dinero.sum(), total), "El dinero total no se conservó"
    return {"dinero": dinero, "t": np.array(tiempos), "S": np.array(entropias),
            "gini": np.array(ginis),
            "tray": {k: np.array(v) for k, v in trayectorias.items()}}


def entropia(dinero, m0):
    """Entropía S = -sum P ln P de la distribución del dinero.

    Para dinero continuo se agrupa en intervalos de ancho m0/20.
    """
    if np.issubdtype(np.asarray(dinero).dtype, np.integer):
        conteos = np.bincount(dinero)
    else:
        conteos = np.bincount((dinero / (m0 / 20)).astype(int))
    p = conteos[conteos > 0] / len(dinero)
    return -np.sum(p * np.log(p))


def gini(dinero):
    """Coeficiente de Gini: 0 = todos iguales; 1 = una persona tiene todo."""
    x = np.sort(np.asarray(dinero, dtype=float))
    n = len(x)
    i = np.arange(1, n + 1)
    return 2 * np.sum(i * x) / (n * x.sum()) - (n + 1) / n


def lorenz(dinero):
    """Curva de Lorenz: fracción acumulada de personas vs fracción acumulada del dinero."""
    x = np.sort(np.asarray(dinero, dtype=float))
    L = np.concatenate([[0], np.cumsum(x) / x.sum()])
    p = np.linspace(0, 1, len(L))
    return p, L


def lorenz_exponencial(p):
    """Curva de Lorenz exacta de la distribución exponencial: L = p + (1-p) ln(1-p)."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = p + (1 - p) * np.log(1 - p)
    return np.where(p >= 1, 1.0, L)


def boltzmann_discreta(m, T, dm=1):
    """P(m) de Boltzmann para dinero en monedas enteras: (1-q) q^m, con q = e^{-dm/T}."""
    q = np.exp(-dm / T)
    return (1 - q) * q ** (np.asarray(m) / dm)


def formas(N, M):
    """Número de formas de repartir M monedas idénticas entre N personas distinguibles."""
    if N == 0:
        return 1 if M == 0 else 0
    return comb(M + N - 1, N - 1)


def prob_exacta(N, M):
    """Probabilidad exacta de que una persona tenga m monedas, contando todos los repartos."""
    total = formas(N, M)
    return np.array([formas(N - 1, M - m) / total for m in range(M + 1)])
