"""
Figuras sencillas para el documento modular. Datos SIMULADOS salvo la curva teórica.

  fig_boltzmann.png   (a) distribución de Boltzmann teórica (monedas enteras)
                      (b) distribución obtenida con el modelo aleatorio contra la teórica
  fig_gini_tiempo.png Gini contra número de intercambios (cuándo se llega al equilibrio)

Uso:  python figuras_documento.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import modelo

AQUI = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(AQUI, "figuras")

N, M0 = 2000, 20
POR_PERSONA = 10_000
SEMILLAS = list(range(200, 240))       # las mismas 40 repeticiones del caso base
MEDIR = 100                            # intercambios por persona entre mediciones
T_EQ = 3000                            # tiempo de equilibrio (convergencia.py)
AZUL, ROJO, GRIS = "#3a6ea5", "#c0504d", "#9bb7d4"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def correr():
    finales, ginis = [], []
    for s in SEMILLAS:
        r = modelo.simular(N, M0, pasos=POR_PERSONA * N, cada=MEDIR * N, semilla=s)
        finales.append(r["dinero"])
        ginis.append(r["gini"])
    return np.concatenate(finales), r["t"], np.array(ginis)


def fig_boltzmann(dinero):
    m = np.arange(0, 121)
    p_teo = modelo.boltzmann_discreta(m, M0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)

    # (a) teoría
    abajo = m < M0
    ax[0].fill_between(m[abajo], p_teo[abajo], step="mid", color=GRIS, alpha=0.7,
                       label=f"Menos que el promedio: {p_teo[abajo].sum():.0%} de las personas")
    ax[0].plot(m, p_teo, "k-", lw=2, label=r"Boltzmann: $P(m)=(1-q)\,q^m$")
    ax[0].axvline(M0, color=ROJO, ls="--", lw=1.5, label=f"Promedio ({M0} monedas)")
    ax[0].set(title="(a) Distribución de Boltzmann (teoría)", xlabel="Monedas de una persona, m",
              ylabel="Fracción de personas")
    ax[0].legend(fontsize=9)

    # (b) simulación contra teoría
    p_sim = np.bincount(dinero, minlength=m.max() + 1)[: m.max() + 1] / len(dinero)
    ax[1].bar(m, p_sim, width=1, color=AZUL, alpha=0.6, label="Simulación (modelo aleatorio)")
    ax[1].plot(m, p_teo, "k-", lw=2, label="Boltzmann (teoría)")
    ax[1].axvline(M0, color=ROJO, ls="--", lw=1.5, label="Todos empezaron aquí")
    ax[1].set(title="(b) Resultado del modelo aleatorio", xlabel="Monedas de una persona, m")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_boltzmann.png"), dpi=150)
    plt.close(fig)


def fig_gini_tiempo(t, ginis):
    g_m = ginis.mean(0)
    g_e = ginis.std(0, ddof=1) / np.sqrt(len(SEMILLAS))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.fill_between(t, g_m - 2 * g_e, g_m + 2 * g_e, color=ROJO, alpha=0.25, lw=0)
    ax.plot(t, g_m, color=ROJO, lw=2, label="Simulación (promedio de 40)")
    ax.axhline(modelo.gini_teorico(M0), color="k", ls="--",
               label=f"Teoría: G = {modelo.gini_teorico(M0):.3f}")
    ax.axvline(T_EQ, color=AZUL, ls=":", lw=2, label=f"Equilibrio (≈ {T_EQ})")
    ax.set(xlabel="Intercambios por persona", ylabel="Coeficiente de Gini", ylim=(0, 0.56),
           xlim=(0, POR_PERSONA))
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_gini_tiempo.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    dinero, t, ginis = correr()
    fig_boltzmann(dinero)
    fig_gini_tiempo(t, ginis)
    print(f"Figuras listas ({len(dinero)} personas en total, datos simulados).")
