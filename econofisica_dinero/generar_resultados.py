"""
Genera todas las figuras y tablas del documento modular.

Todos los resultados son de DATOS SIMULADOS con el modelo de Drăgulescu y Yakovenko (2000).

Uso:  python generar_resultados.py
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

import modelo

AQUI = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(AQUI, "figuras")
RES = os.path.join(AQUI, "resultados")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

AZUL, ROJO, GRIS = "#3a6ea5", "#c0504d", "#9bb7d4"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# Caso base
N, M0, DM, PASOS, SEMILLA = 2000, 20, 1, 4_000_000, 42


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, nombre), dpi=150)
    plt.close(fig)


def fig_conteo():
    """Conteo exacto: sin simular nada, solo contando repartos."""
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    casos = [(3, 3), (10, 50), (100, 1000)]
    for k, (n, m) in enumerate(casos):
        p = modelo.prob_exacta(n, m)
        x = np.arange(len(p))
        if k == 0:
            ax[k].bar(x, p, color=AZUL)
            for xi, pi in zip(x, p):
                ax[k].text(xi, pi + 0.01, f"{pi:.0%}", ha="center")
        else:
            lim = min(len(p), int(6 * m / n))
            ax[k].bar(x[:lim], p[:lim], width=1, color=GRIS, label="Conteo exacto")
            T = m / n
            ax[k].plot(x[:lim], np.exp(-x[:lim] / T) / T, "k-", lw=2, label=r"$e^{-m/T}/T$")
            ax[k].legend()
        ax[k].axvline(m / n, color=ROJO, ls="--")
        ax[k].set(title=f"{n} personas, {m} monedas", xlabel="Monedas de una persona, m",
                  ylabel="Probabilidad" if k == 0 else "")
    guardar(fig, "fig1_conteo_exacto.png")


def fig_distribucion(r):
    d = r["dinero"]
    m = np.arange(d.max() + 1)
    p_sim = np.bincount(d, minlength=len(m)) / N
    p_teo = modelo.boltzmann_discreta(m, M0, DM)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(m, p_sim, width=1, color=GRIS, label="Simulación")
    ax[0].plot(m, p_teo, "k-", lw=2, label=r"Boltzmann $e^{-m/T}/T$")
    ax[0].axvline(M0, color=ROJO, ls="--", label=f"Dinero inicial de todos ({M0})")
    ax[0].set(xlabel="Dinero m", ylabel="Fracción de personas P(m)")
    ax[0].legend()
    ax[1].semilogy(m, np.where(p_sim > 0, p_sim, np.nan), "o", ms=4, color=AZUL, label="Simulación")
    ax[1].semilogy(m, p_teo, "k-", lw=2, label="Boltzmann (recta)")
    ax[1].set(xlabel="Dinero m", ylabel="P(m), escala logarítmica", ylim=(1e-4, 0.1))
    ax[1].legend()
    guardar(fig, "fig2_distribucion.png")


def fig_evolucion(r):
    m = np.arange(400)
    p = modelo.boltzmann_discreta(m, M0, DM)
    s_max = -np.sum(p * np.log(p))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(r["t"], r["S"], color=AZUL)
    ax[0].axhline(s_max, color="k", ls="--", label=f"Máximo teórico ({s_max:.2f})")
    ax[0].set(xlabel="Intercambios por persona", ylabel="Entropía S")
    ax[0].legend(loc="lower right")
    ax[1].plot(r["t"], r["gini"], color=ROJO)
    ax[1].axhline(0.5, color="k", ls="--", label="Teoría: G = 0.5")
    ax[1].set(xlabel="Intercambios por persona", ylabel="Coeficiente de Gini G")
    ax[1].legend(loc="lower right")
    guardar(fig, "fig3_evolucion.png")
    return s_max


def fig_lorenz(r):
    p, L = modelo.lorenz(r["dinero"])
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], color="0.5", ls=":", label="Igualdad perfecta (inicio)")
    ax.plot(p, L, color=AZUL, lw=3, label="Simulación (final)")
    ax.plot(p, modelo.lorenz_exponencial(p), "k--", label="Exponencial exacta")
    ax.fill_between(p, L, p, color=GRIS, alpha=0.4)
    ax.set(xlabel="Fracción de personas (de más pobre a más rica)",
           ylabel="Fracción del dinero total", aspect="equal")
    ax.legend(loc="upper left", fontsize=9)
    guardar(fig, "fig4_lorenz.png")
    i50 = len(p) // 2
    i90 = int(0.9 * (len(p) - 1))
    return L[i50], 1 - L[i90]


def fig_movilidad(r):
    tray = r["tray"]
    t = r["t"]
    eq = t > 500  # después del equilibrio
    frac_abajo = np.array([np.mean(v[eq] < M0) for v in tray.values()])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, c in zip(list(tray)[:3], [AZUL, ROJO, "#6a9a4f"]):
        ax[0].plot(t, tray[k], color=c, lw=1, label=f"Persona {k}")
    ax[0].axhline(M0, color="k", ls="--", lw=1, label="Promedio")
    ax[0].set(xlabel="Intercambios por persona", ylabel="Dinero")
    ax[0].legend(fontsize=9, ncol=2)
    ax[1].hist(frac_abajo, bins=30, color=GRIS, edgecolor="white")
    ax[1].axvline(1 - np.exp(-1), color="k", ls="--", label=r"Teoría $1-e^{-1}\approx0.63$")
    ax[1].set(xlabel="Fracción del tiempo con menos que el promedio",
              ylabel="Número de personas")
    ax[1].legend(fontsize=9)
    guardar(fig, "fig5_movilidad.png")
    return frac_abajo.mean(), frac_abajo.std()


def robustez():
    """Cambia los parámetros y la regla de intercambio; mide G y P(m < promedio)."""
    configs = [
        ("Base", dict(N=2000, m0=20, dm=1)),
        ("Menos personas", dict(N=500, m0=20, dm=1)),
        ("Más pobre (m0=10)", dict(N=2000, m0=10, dm=1)),
        ("Más rica (m0=40, dm=2)", dict(N=2000, m0=40, dm=2)),
        ("Regla de reparto al azar", dict(N=2000, m0=20, regla="reparto")),
    ]
    filas = []
    for nombre, c in configs:
        gs, fs = [], []
        for s in range(5):
            r = modelo.simular(pasos=4_000_000, cada=4_000_000, semilla=100 + s, **c)
            gs.append(modelo.gini(r["dinero"]))
            fs.append(np.mean(r["dinero"] < c["m0"]))
        # Gini teórico: 1/(1+e^{-dm/T}) con monedas discretas; 1/2 si el dinero es continuo
        g_teo = 1 / (1 + np.exp(-c["dm"] / c["m0"])) if "dm" in c else 0.5
        filas.append([nombre, c["N"], c["m0"], c.get("dm", "-"),
                      f"{np.mean(gs):.3f}", f"{np.std(gs):.3f}", f"{g_teo:.3f}",
                      f"{np.mean(fs):.3f}"])
        print(f"  {nombre}: G = {np.mean(gs):.3f} ± {np.std(gs):.3f} (teoría {g_teo:.3f})")
    with open(os.path.join(RES, "robustez.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["configuracion", "N", "m0", "dm", "gini_media", "gini_desv", "gini_teorico", "frac_bajo_promedio"])
        w.writerows(filas)
    return filas


if __name__ == "__main__":
    print("Datos SIMULADOS (modelo de Drăgulescu-Yakovenko).")
    fig_conteo()
    r = modelo.simular(N, M0, DM, pasos=PASOS, cada=10_000, semilla=SEMILLA,
                       seguir=range(N))
    fig_distribucion(r)
    s_max = fig_evolucion(r)
    mitad, top10 = fig_lorenz(r)
    fm, fs = fig_movilidad(r)
    d = r["dinero"]
    resumen = {
        "N": N, "m0": M0, "dm": DM, "pasos": PASOS,
        "dinero_total": int(d.sum()),
        "gini_final": round(modelo.gini(d), 3),
        "entropia_final": round(r["S"][-1], 3), "entropia_max": round(s_max, 3),
        "frac_bajo_promedio": round(np.mean(d < M0), 3),
        "frac_sin_dinero": round(np.mean(d == 0), 3),
        "dinero_mitad_pobre": round(mitad, 3), "dinero_10pct_rico": round(top10, 3),
        "max_dinero": int(d.max()),
        "frac_tiempo_bajo_promedio_media": round(fm, 3),
        "frac_tiempo_bajo_promedio_desv": round(fs, 3),
    }
    with open(os.path.join(RES, "resumen_caso_base.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cantidad", "valor"])
        w.writerows(resumen.items())
    for k, v in resumen.items():
        print(f"  {k}: {v}")
    print("Robustez (5 semillas por configuración):")
    robustez()
