"""
Extensión del modelo: se permite endeudarse hasta un límite m_d
(Drăgulescu y Yakovenko, 2000, sección sobre deuda).

Regla: el pagador puede entregar dm mientras su dinero no baje de -m_d.
Predicción: la misma deducción de Boltzmann con la variable m' = m + m_d >= 0 da una
geométrica en m' con promedio <m> + m_d, es decir q = (<m>+m_d)/(1+<m>+m_d) y
temperatura exacta T = 1/ln(1 + 1/(<m>+m_d)) ≈ <m> + m_d + 1/2. El Gini exacto es
G = (1 + m_d/<m>)/(1 + q), que en el límite continuo es (1 + m_d/<m>)/2.

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
POR_PERSONA = 20_000           # con deuda la temperatura es mayor y el equilibrio tarda más
PASOS = POR_PERSONA * N        # (el tiempo de relajación crece aprox. como T^2; ver convergencia.py)
SEMILLAS = list(range(11, 31)) # 20 repeticiones
COLORES = ["0.3", "#3a6ea5", "#d98b3a", "#c0504d"]


def teoria(md):
    """Distribución exacta (geométrica desplazada), Gini y fracción en deuda."""
    m = np.arange(-md, int(10 * (M0 + md)))
    p = modelo.boltzmann_discreta(m, M0, md)
    return m, p, modelo.gini_teorico(M0, md), modelo.participaciones_teoricas(M0, md)["en_deuda"]


def principal():
    print("Datos SIMULADOS: modelo con deuda.")
    filas, finales = [], {}
    for md in LIMITES:
        med = []
        for sem in SEMILLAS:
            d = modelo.simular(N, M0, DM, pasos=PASOS, cada=PASOS, semilla=sem, deuda=md)["dinero"]
            assert d.sum() == N * M0 and d.min() >= -md
            x = np.sort(d)
            med.append((modelo.gini(d), np.mean(d < 0), x[int(0.9 * N):].sum() / x.sum(),
                        x[: N // 2].sum() / x.sum()))
            finales.setdefault(md, []).append(d)
        med = np.array(med)
        g, g_d, g_e = modelo.media_y_error(med[:, 0])
        teo = modelo.participaciones_teoricas(M0, md)
        _, _, g_teo, _ = teoria(md)
        T = modelo.temperatura_exacta(M0 + md)
        filas.append([md, f"{T:.2f}", f"{g:.4f}", f"{g_e:.4f}", f"{g_d:.4f}", f"{g_teo:.4f}",
                      f"{modelo.gini_teorico(M0, md, continuo=True):.4f}",
                      f"{100 * med[:, 1].mean():.1f}", f"{100 * teo['en_deuda']:.1f}",
                      f"{100 * med[:, 2].mean():.1f}", f"{100 * teo['diez_rico']:.1f}",
                      f"{100 * med[:, 3].mean():.1f}", f"{100 * teo['mitad_pobre']:.1f}"])
        print(f"  m_d={md:2d}: T={T:.2f}, Gini {g:.4f} ± {g_e:.4f} (teoría exacta {g_teo:.4f}, "
              f"continua {filas[-1][6]}; {(g - g_teo) / g_e:+.1f} σ), en deuda {filas[-1][7]} % "
              f"(teoría {filas[-1][8]} %), 10% rico {filas[-1][9]} % ({filas[-1][10]} %), "
              f"50% pobre {filas[-1][11]} % ({filas[-1][12]} %)")
    with open(os.path.join(RES, "deudas.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["limite_deuda", "temperatura_exacta", "gini_media", "gini_error_estandar",
                    "gini_desv", "gini_teorico_exacto", "gini_limite_continuo",
                    "pct_en_deuda", "pct_en_deuda_teorico", "pct_10_rico", "pct_10_rico_teorico",
                    "pct_50_pobre", "pct_50_pobre_teorico"])
        w.writerows(filas)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for md, c in zip(LIMITES, COLORES):
        d = np.concatenate(finales[md])
        m = np.arange(d.min(), d.max() + 1)
        p = np.bincount(d - d.min()) / len(d)
        ax[0].semilogy(m, np.where(p > 0, p, np.nan), "o", ms=2.5, color=c, alpha=0.6)
        mt, pt, _, _ = teoria(md)
        ax[0].semilogy(mt, pt, "-", color=c, lw=2, label=f"$m_d$ = {md}  (T = {modelo.temperatura_exacta(M0 + md):.1f})")
    ax[0].axvline(0, color="k", lw=0.8, ls=":")
    ax[0].set(xlabel="Dinero m (negativo = deuda)", ylabel="P(m), escala logarítmica",
              ylim=(1e-5, 0.1), xlim=(-22, 160),
              title="Distribución: puntos = simulación, líneas = teoría")
    ax[0].legend(fontsize=9)

    x = np.linspace(0, 22, 100)
    ax[1].plot(x, 0.5 * (1 + x / M0), "k-", label=r"Teoría continua $G=\frac{1}{2}(1+m_d/\langle m\rangle)$")
    ax[1].plot(LIMITES, [float(f[5]) for f in filas], "ks", mfc="white", label="Teoría exacta (monedas enteras)")
    ax[1].errorbar(LIMITES, [float(f[2]) for f in filas], yerr=[float(f[3]) for f in filas], fmt="o",
                   color="#c0504d", capsize=3, label="Simulación (media ± error estándar, 20 semillas)")
    ax[1].set(xlabel="Límite de deuda $m_d$ (dinero promedio = 20)", ylabel="Coeficiente de Gini",
              title="Más crédito, más desigualdad")
    ax[1].legend(fontsize=9)
    fig.suptitle("Datos simulados: modelo con deuda (N = 2000, dinero promedio 20)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig8_deudas.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    principal()
