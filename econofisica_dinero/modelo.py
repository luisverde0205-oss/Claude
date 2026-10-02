"""
Núcleo del modelo de Drăgulescu y Yakovenko (2000), funciones de medición y
predicciones teóricas EXACTAS para dinero en monedas enteras.

Todos los datos que produce este módulo son SIMULADOS: no provienen de ninguna
encuesta ni registro real.

Notación: el dinero se mide en monedas (dm = 1). mu = <m> = M/N es el dinero promedio.
En equilibrio la distribución es geométrica, P(m) = (1-q) q^m, con q = mu/(1+mu).
La temperatura exacta es T = -1/ln q = 1/ln(1 + 1/mu) ≈ mu + 1/2; en el límite
continuo (mu >> 1) se recupera T = mu y la exponencial e^{-m/T}/T.
"""
from math import comb

import numpy as np
from numba import njit


@njit(cache=True)
def _intercambios_fijo(dinero, a, b, dm, deuda, ini, fin):
    for t in range(ini, fin):
        i, j = a[t], b[t]
        if i != j and dinero[i] - dm >= -deuda:
            dinero[i] -= dm
            dinero[j] += dm


@njit(cache=True)
def _intercambios_reparto(dinero, a, b, eps, ini, fin):
    for t in range(ini, fin):
        i, j = a[t], b[t]
        if i != j:
            s = dinero[i] + dinero[j]
            dinero[i] = eps[t] * s
            dinero[j] = s - dinero[i]


def simular(N, m0, dm=1, pasos=4_000_000, cada=20_000, semilla=0,
            regla="fijo", seguir=(), deuda=0):
    """Simula N personas que intercambian dinero al azar.

    regla="fijo":   el pagador entrega `dm` monedas; si no las tiene, no hay intercambio.
    regla="reparto": la pareja junta su dinero y lo reparte con una fracción al azar
                     (variante continua, también conserva el dinero).
    seguir:         índices de personas cuyo dinero se registra a lo largo del tiempo.
    deuda:          límite de deuda m_d (solo regla "fijo"): se puede pagar mientras
                    el dinero no baje de -m_d. Con deuda=0 es el modelo original.

    Ambas reglas son simétricas: pasar de un reparto A a uno B tiene la misma
    probabilidad que pasar de B a A, por lo que en equilibrio todos los repartos
    accesibles son igualmente probables.

    Devuelve un diccionario con el dinero final y las series de tiempo medidas
    (se mide después del paso t, para t = 0, cada, 2*cada, ...).
    """
    rng = np.random.default_rng(semilla)
    entero = regla == "fijo"
    dinero = np.full(N, m0, dtype=np.int64 if entero else float)
    total = dinero.sum()
    a = rng.integers(0, N, pasos)
    b = rng.integers(0, N, pasos)
    eps = rng.random(pasos) if regla == "reparto" else None

    def avanzar(ini, fin):
        if entero:
            _intercambios_fijo(dinero, a, b, dm, deuda, ini, fin)
        else:
            _intercambios_reparto(dinero, a, b, eps, ini, fin)

    tiempos, entropias, ginis = [], [], []
    trayectorias = {k: [] for k in seguir}
    for t0 in range(0, pasos, cada):
        avanzar(t0, t0 + 1)
        tiempos.append(t0 / N)
        entropias.append(entropia(dinero, m0))
        ginis.append(gini(dinero))
        for k in seguir:
            trayectorias[k].append(dinero[k])
        avanzar(t0 + 1, min(t0 + cada, pasos))

    assert np.isclose(dinero.sum(), total), "El dinero total no se conservó"
    return {"dinero": dinero, "t": np.array(tiempos), "S": np.array(entropias),
            "gini": np.array(ginis),
            "tray": {k: np.array(v) for k, v in trayectorias.items()}}


# ---------------------------------------------------------------- mediciones

def entropia(dinero, m0, corregida=True):
    """Entropía S = -sum P ln P de la distribución del dinero.

    El estimador directo (con las frecuencias del histograma) subestima la entropía
    cuando la muestra es finita; con corregida=True se suma la corrección de
    Miller-Madow, (K - 1) / (2 n), con K = número de valores ocupados.
    Para dinero continuo se agrupa en intervalos de ancho m0/20.
    """
    if np.issubdtype(np.asarray(dinero).dtype, np.integer):
        conteos = np.bincount(dinero - dinero.min())
    else:
        conteos = np.bincount((dinero / (m0 / 20)).astype(int))
    conteos = conteos[conteos > 0]
    p = conteos / len(dinero)
    s = -np.sum(p * np.log(p))
    if corregida:
        s += (len(conteos) - 1) / (2 * len(dinero))
    return s


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


def media_y_error(valores):
    """Media, desviación estándar entre semillas y error estándar de la media."""
    v = np.asarray(valores, dtype=float)
    d = v.std(ddof=1) if len(v) > 1 else 0.0
    return v.mean(), d, d / np.sqrt(len(v))


# ------------------------------------------------ teoría exacta (monedas enteras)

def q_exacto(mu):
    """Razón de la distribución geométrica con promedio mu: q = mu/(1+mu) = e^{-1/T}."""
    return mu / (1 + mu)


def temperatura_exacta(mu):
    """T = 1/ln(1 + 1/mu), en monedas. Para mu >> 1, T ≈ mu + 1/2."""
    return 1 / np.log1p(1 / mu)


def boltzmann_discreta(m, mu, deuda=0):
    """P(m) exacta en equilibrio: geométrica en m' = m + deuda con promedio mu + deuda."""
    q = q_exacto(mu + deuda)
    m = np.asarray(m)
    return np.where(m >= -deuda, (1 - q) * q ** (m + deuda), 0.0)


def gini_teorico(mu, deuda=0, continuo=False):
    """Gini de equilibrio.

    Exacto (monedas):  G = (1 + deuda/mu) / (1 + q),  q = (mu + deuda)/(1 + mu + deuda).
    Continuo:          G = (1 + deuda/mu) / 2.
    """
    if continuo:
        return 0.5 * (1 + deuda / mu)
    return (1 + deuda / mu) / (1 + q_exacto(mu + deuda))


def entropia_teorica(mu, continuo=False):
    """Entropía de equilibrio por persona: geométrica (1+mu)ln(1+mu) - mu ln mu; continua 1 + ln mu."""
    if continuo:
        return 1 + np.log(mu)
    return (1 + mu) * np.log(1 + mu) - mu * np.log(mu)


def participaciones_teoricas(mu, deuda=0, mmax_factor=40):
    """Predicciones exactas a partir de la geométrica: fracciones de población y de dinero."""
    m = np.arange(-deuda, int(mmax_factor * (mu + deuda + 1)))
    p = boltzmann_discreta(m, mu, deuda)
    p /= p.sum()
    F = np.cumsum(p)                       # fracción de personas con dinero <= m
    D = np.cumsum(p * m) / np.sum(p * m)   # fracción del dinero que tienen

    def parte_inferior(f):                 # dinero del f más pobre (interpolando dentro de un nivel)
        k = np.searchsorted(F, f)
        previo_F = F[k - 1] if k > 0 else 0.0
        previo_D = D[k - 1] if k > 0 else 0.0
        return previo_D + (f - previo_F) * m[k] / np.sum(p * m)

    return {"bajo_promedio": float(p[m < mu].sum()),
            "sin_dinero": float(p[m == 0].sum()),
            "en_deuda": float(p[m < 0].sum()),
            "mitad_pobre": float(parte_inferior(0.5)),
            "diez_rico": float(1 - parte_inferior(0.9))}


def lorenz_exponencial(p):
    """Curva de Lorenz de la distribución exponencial (límite continuo): L = p + (1-p) ln(1-p)."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = p + (1 - p) * np.log(1 - p)
    return np.where(p >= 1, 1.0, L)


# ------------------------------------------- ensamble exacto (todos los repartos igual de probables)

def muestra_microcanonica(N, M, rng, continuo=False):
    """Un reparto escogido uniformemente entre TODOS los repartos de M entre N personas.

    Monedas: se colocan N-1 separadores al azar entre M+N-1 posiciones ("estrellas y barras").
    Continuo: punto uniforme del símplex (exponenciales normalizadas).
    """
    if continuo:
        e = rng.exponential(size=N)
        return M * e / e.sum()
    b = np.sort(rng.choice(M + N - 1, N - 1, replace=False))
    return np.diff(np.concatenate([[-1], b, [M + N - 1]])) - 1


def estadisticas(dinero, m0):
    """Cantidades que se comparan con la teoría (fracciones entre 0 y 1)."""
    x = np.sort(dinero)
    n = len(x)
    return {"gini": gini(x), "entropia": entropia(x, m0), "bajo_promedio": np.mean(x < m0),
            "sin_dinero": np.mean(x == 0), "mitad_pobre": x[: n // 2].sum() / x.sum(),
            "diez_rico": x[int(0.9 * n):].sum() / x.sum()}


def ensamble_esperado(N, m0, muestras=2000, semilla=12345, continuo=False):
    """Media y desviación estándar de cada estadística sobre repartos del ensamble exacto.

    Es la predicción teórica para un sistema FINITO de N personas, medida con los mismos
    estimadores que la simulación (incluye el sesgo de muestra finita de la entropía).
    """
    rng = np.random.default_rng(semilla)
    valores = [estadisticas(muestra_microcanonica(N, N * m0, rng, continuo), m0)
               for _ in range(muestras)]
    return {k: (np.mean([v[k] for v in valores]), np.std([v[k] for v in valores], ddof=1))
            for k in valores[0]}


# ------------------------------------------------------------ conteo exacto

def formas(N, M):
    """Número de formas de repartir M monedas idénticas entre N personas distinguibles."""
    if N == 0:
        return 1 if M == 0 else 0
    return comb(M + N - 1, N - 1)


def prob_exacta(N, M):
    """Probabilidad exacta de que una persona tenga m monedas, contando todos los repartos."""
    total = formas(N, M)
    return np.array([formas(N - 1, M - m) / total for m in range(M + 1)])
