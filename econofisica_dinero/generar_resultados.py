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

# Caso base: 10 000 intercambios por persona (el análisis de convergencia, convergencia.py,
# muestra que el Gini se estabiliza después de unos 5000).
N, M0, DM, SEMILLA = 2000, 20, 1, 42
POR_PERSONA = 10_000
PASOS = POR_PERSONA * N
SEMILLAS_BASE = list(range(200, 240))   # 40 repeticiones del caso base
SEMILLAS = list(range(100, 120))        # 20 repeticiones por configuración de robustez


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
            ax[k].plot(x[:lim], modelo.boltzmann_discreta(x[:lim], m / n), "k-", lw=2,
                       label="Geométrica (N grande)")
            ax[k].legend()
        ax[k].axvline(m / n, color=ROJO, ls="--")
        ax[k].set(title=f"{n} personas, {m} monedas", xlabel="Monedas de una persona, m",
                  ylabel="Probabilidad" if k == 0 else "")
    guardar(fig, "fig1_conteo_exacto.png")


def fig_distribucion(r):
    d = r["dinero"]
    m = np.arange(d.max() + 1)
    p_sim = np.bincount(d, minlength=len(m)) / N
    p_teo = modelo.boltzmann_discreta(m, M0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(m, p_sim, width=1, color=GRIS, label="Simulación")
    ax[0].plot(m, p_teo, "k-", lw=2, label=r"Boltzmann: $(1-q)\,q^m$")
    ax[0].axvline(M0, color=ROJO, ls="--", label=f"Dinero inicial de todos ({M0})")
    ax[0].set(xlabel="Dinero m", ylabel="Fracción de personas P(m)")
    ax[0].legend()
    ax[1].semilogy(m, np.where(p_sim > 0, p_sim, np.nan), "o", ms=4, color=AZUL, label="Simulación")
    ax[1].semilogy(m, p_teo, "k-", lw=2, label="Boltzmann (recta)")
    ax[1].set(xlabel="Dinero m", ylabel="P(m), escala logarítmica", ylim=(1e-4, 0.1))
    ax[1].legend()
    guardar(fig, "fig2_distribucion.png")


def fig_evolucion(r):
    s_max = modelo.entropia_teorica(M0)
    g_teo = modelo.gini_teorico(M0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(r["t"], r["S"], color=AZUL)
    ax[0].axhline(s_max, color="k", ls="--", label=f"Máximo teórico exacto ({s_max:.3f})")
    ax[0].set(xlabel="Intercambios por persona", ylabel="Entropía S", xscale="symlog",
              xlim=(0, POR_PERSONA))
    ax[0].legend(loc="lower right")
    ax[1].plot(r["t"], r["gini"], color=ROJO)
    ax[1].axhline(g_teo, color="k", ls="--", label=f"Teoría exacta: G = {g_teo:.3f}")
    ax[1].axhline(0.5, color="0.5", ls=":", label="Límite continuo: G = 1/2")
    ax[1].set(xscale="symlog", xlim=(0, POR_PERSONA))
    ax[1].set(xlabel="Intercambios por persona", ylabel="Coeficiente de Gini G")
    ax[1].legend(loc="lower right")
    guardar(fig, "fig3_evolucion.png")
    return s_max


def fig_lorenz(r):
    p, L = modelo.lorenz(r["dinero"])
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], color="0.5", ls=":", label="Igualdad perfecta (inicio)")
    ax.plot(p, L, color=AZUL, lw=3, label="Simulación (final)")
    ax.plot(p, modelo.lorenz_exponencial(p), "k--", label="Exponencial (límite continuo)")
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
    eq = t > 5000  # después de alcanzar el equilibrio (ver convergencia.py)
    frac_abajo = np.array([np.mean(v[eq] < M0) for v in tray.values()])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, c in zip(list(tray)[:3], [AZUL, ROJO, "#6a9a4f"]):
        ax[0].plot(t, tray[k], color=c, lw=1, label=f"Persona {k}")
    ax[0].axhline(M0, color="k", ls="--", lw=1, label="Promedio")
    ax[0].set(xlabel="Intercambios por persona", ylabel="Dinero")
    ax[0].legend(fontsize=9, ncol=2)
    ax[1].hist(frac_abajo, bins=30, color=GRIS, edgecolor="white")
    teo = modelo.participaciones_teoricas(M0)["bajo_promedio"]
    ax[1].axvline(teo, color="k", ls="--", label=f"Teoría exacta: {teo:.3f}")
    ax[1].set(xlabel="Fracción del tiempo con menos que el promedio",
              ylabel="Número de personas")
    ax[1].legend(fontsize=9)
    guardar(fig, "fig5_movilidad.png")
    return frac_abajo.mean(), frac_abajo.std()


def robustez():
    """Cambia los parámetros y la regla de intercambio; mide el Gini de equilibrio.

    20 semillas por configuración, 10 000 intercambios por persona. Se reportan la media,
    la desviación estándar entre semillas, el error estándar de la media, la teoría exacta
    (N infinito) y el valor esperado en el ensamble exacto para el mismo N.
    """
    configs = [
        ("Base", dict(N=2000, m0=20, dm=1)),
        ("Menos personas", dict(N=500, m0=20, dm=1)),
        ("Menos dinero (m0=10)", dict(N=2000, m0=10, dm=1)),
        ("Más dinero y pagos (m0=40, dm=2)", dict(N=2000, m0=40, dm=2)),
        ("Regla de reparto al azar", dict(N=2000, m0=20, regla="reparto")),
    ]
    filas = []
    for nombre, c in configs:
        gs = [modelo.gini(modelo.simular(pasos=POR_PERSONA * c["N"], cada=POR_PERSONA * c["N"],
                                         semilla=s, **c)["dinero"]) for s in SEMILLAS]
        media, desv, err = modelo.media_y_error(gs)
        # Teoría exacta: promedio en unidades del pago, mu = m0/dm; la regla de reparto es continua
        g_teo = modelo.gini_teorico(c["m0"] / c["dm"]) if "dm" in c else 0.5
        ens = modelo.ensamble_esperado(c["N"], c["m0"] // c.get("dm", 1), muestras=1000,
                                       continuo="dm" not in c)["gini"]
        z = (media - ens[0]) / np.hypot(err, ens[1] / np.sqrt(1000))
        filas.append([nombre, c["N"], c["m0"], c.get("dm", "-"), f"{media:.4f}", f"{desv:.4f}",
                      f"{err:.4f}", f"{g_teo:.4f}", f"{ens[0]:.4f}", f"{ens[1]:.4f}", f"{z:+.1f}"])
        print(f"  {nombre}: G = {media:.4f} ± {err:.4f} (desv. {desv:.4f}); teoría {g_teo:.4f}; "
              f"ensamble N={c['N']}: {ens[0]:.4f} (desv. {ens[1]:.4f}); diferencia {z:+.1f} σ")
    with open(os.path.join(RES, "robustez.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["configuracion", "N", "m0", "dm", "gini_media", "gini_desv", "gini_error_estandar",
                    "gini_teorico_N_infinito", "gini_ensamble_mismo_N", "desv_ensamble",
                    "diferencia_en_errores_estandar"])
        w.writerows(filas)
    return filas


def tabla_base():
    """Caso base con 40 semillas comparado con la teoría exacta, el ensamble exacto
    para N = 2000 (misma forma de medir) y el límite continuo."""
    medidas = [modelo.estadisticas(modelo.simular(N, M0, DM, pasos=PASOS, cada=PASOS,
                                                  semilla=s)["dinero"], M0)
               for s in SEMILLAS_BASE]
    ens = modelo.ensamble_esperado(N, M0, muestras=2000)
    teo = modelo.participaciones_teoricas(M0)
    teo.update(gini=modelo.gini_teorico(M0), entropia=modelo.entropia_teorica(M0))
    cont = {"gini": 0.5, "entropia": modelo.entropia_teorica(M0, continuo=True),
            "bajo_promedio": 1 - np.exp(-1), "sin_dinero": 1 - np.exp(-1 / M0),
            "mitad_pobre": float(modelo.lorenz_exponencial(0.5)),
            "diez_rico": float(1 - modelo.lorenz_exponencial(0.9))}
    filas = []
    for k in teo:
        if k == "en_deuda":
            continue
        media, desv, err = modelo.media_y_error([m[k] for m in medidas])
        z = (media - ens[k][0]) / np.hypot(err, ens[k][1] / np.sqrt(2000))
        filas.append([k, f"{media:.4f}", f"{err:.4f}", f"{desv:.4f}", f"{ens[k][0]:.4f}",
                      f"{ens[k][1]:.4f}", f"{teo[k]:.4f}", f"{cont[k]:.4f}", f"{z:+.1f}"])
        print(f"  {k}: {media:.4f} ± {err:.4f} (desv. {desv:.4f}); ensamble N=2000 "
              f"{ens[k][0]:.4f} (desv. {ens[k][1]:.4f}); N infinito {teo[k]:.4f}; "
              f"continuo {cont[k]:.4f}; diferencia {z:+.1f} σ")
    with open(os.path.join(RES, "tabla_caso_base.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cantidad", "media_40_semillas", "error_estandar", "desv_entre_semillas",
                    "ensamble_exacto_N2000", "desv_ensamble", "teoria_N_infinito",
                    "limite_continuo", "diferencia_en_errores_estandar"])
        w.writerows(filas)


if __name__ == "__main__":
    print("Datos SIMULADOS (modelo de Drăgulescu-Yakovenko).")
    fig_conteo()
    r = modelo.simular(N, M0, DM, pasos=PASOS, cada=10_000, semilla=SEMILLA,
                       seguir=range(N))
    fig_distribucion(r)
    s_max = fig_evolucion(r)
    print("Caso base, 40 semillas:")
    tabla_base()
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
    print("Robustez (20 semillas por configuración):")
    robustez()
