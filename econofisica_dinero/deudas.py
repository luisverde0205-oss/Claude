"""
Extensión del modelo: se permite endeudarse hasta un límite m_d
(Drăgulescu y Yakovenko, 2000, sección sobre deuda).

Regla: el pagador puede entregar dm mientras su dinero no baje de -m_d.
Predicción: la misma deducción de Boltzmann con la variable m' = m + m_d >= 0 da
    P(m) ∝ exp(-(m + m_d)/T),   T = M/N + m_d,
y el coeficiente de Gini (límite continuo) G = (1 + m_d/<m>)/2.

Todos los resultados son DATOS SIMULADOS.

Uso:  python deudas.py
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

import modelo

AQUI = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(AQUI, "figuras")
RES = os.path.join(AQUI, "resultados")

N, M0, DM = 2000, 20, 1
LIMITES = [0, 5, 10, 20]       # límite de deuda m_d
PASOS = 10_000_000             # 5000 intercambios por persona (con deuda el equilibrio tarda más)
SEMILLAS = [11, 12, 13]
COLORES = ["0.3", "#3a6ea5", "#d98b3a", "#c0504d"]


def teoria(md):
    """Distribución geométrica desplazada (monedas enteras) y su Gini exacto."""
    Tp = M0 + md                         # promedio de m' = m + m_d
    q = Tp / (1 + Tp)                    # q = e^{-1/T} con el promedio exacto
    m = np.arange(-md, int(10 * Tp))
    p = (1 - q) * q ** (m + md)
    dif_media = 2 * q / ((1 - q) * (1 + q))   # E|X - Y| de dos geométricas
    gini = dif_media / (2 * M0)
    en_deuda = 1 - q ** md               # P(m < 0)
    return m, p, gini, en_deuda


def principal():
    print("Datos SIMULADOS: modelo con deuda.")
    filas, finales = [], {}
    for md in LIMITES:
        g, frac, tops, pobres = [], [], [], []
        for s in SEMILLAS:
            d = modelo.simular(N, M0, DM, pasos=PASOS, cada=PASOS, semilla=s, deuda=md)["dinero"]
            assert d.sum() == N * M0 and d.min() >= -md
            g.append(modelo.gini(d))
            frac.append(np.mean(d < 0))
            x = np.sort(d)
            tops.append(100 * x[int(0.9 * N):].sum() / x.sum())
            pobres.append(100 * x[: N // 2].sum() / x.sum())
            finales.setdefault(md, []).append(d)
        _, _, g_teo, deuda_teo = teoria(md)
        filas.append([md, M0 + md, round(np.mean(g), 3), round(np.std(g), 3), round(g_teo, 3),
                      round(100 * np.mean(frac), 1), round(100 * deuda_teo, 1),
                      round(np.mean(tops), 1), round(np.mean(pobres), 1)])
        print(f"  m_d={md:2d}: T={M0 + md}, Gini {filas[-1][2]} ± {filas[-1][3]} (teoría {filas[-1][4]}), "
              f"en deuda {filas[-1][5]} % (teoría {filas[-1][6]} %), "
              f"10% rico {filas[-1][7]} %, 50% pobre {filas[-1][8]} %")
    with open(os.path.join(RES, "deudas.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["limite_deuda", "temperatura", "gini", "gini_desv", "gini_teorico",
                    "pct_en_deuda", "pct_en_deuda_teorico", "pct_10_rico", "pct_50_pobre"])
        w.writerows(filas)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for md, c in zip(LIMITES, COLORES):
        d = np.concatenate(finales[md])
        m = np.arange(d.min(), d.max() + 1)
        p = np.bincount(d - d.min()) / len(d)
        ax[0].semilogy(m, np.where(p > 0, p, np.nan), "o", ms=2.5, color=c, alpha=0.6)
        mt, pt, _, _ = teoria(md)
        ax[0].semilogy(mt, pt, "-", color=c, lw=2, label=f"$m_d$ = {md}  (T = {M0 + md})")
    ax[0].axvline(0, color="k", lw=0.8, ls=":")
    ax[0].set(xlabel="Dinero m (negativo = deuda)", ylabel="P(m), escala logarítmica",
              ylim=(1e-5, 0.1), xlim=(-22, 160),
              title="Distribución: puntos = simulación, líneas = teoría")
    ax[0].legend(fontsize=9)

    x = np.linspace(0, 22, 100)
    ax[1].plot(x, 0.5 * (1 + x / M0), "k-", label=r"Teoría continua $G=\frac{1}{2}(1+m_d/\langle m\rangle)$")
    ax[1].plot(LIMITES, [f[4] for f in filas], "ks", mfc="white", label="Teoría con monedas enteras")
    ax[1].errorbar(LIMITES, [f[2] for f in filas], yerr=[f[3] for f in filas], fmt="o",
                   color="#c0504d", capsize=3, label="Simulación (3 semillas)")
    ax[1].set(xlabel="Límite de deuda $m_d$ (dinero promedio = 20)", ylabel="Coeficiente de Gini",
              title="Más crédito, más desigualdad")
    ax[1].legend(fontsize=9)
    fig.suptitle("Datos simulados: modelo con deuda (N = 2000, dinero promedio 20)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig8_deudas.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    principal()
