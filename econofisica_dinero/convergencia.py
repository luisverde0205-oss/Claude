"""
Convergencia e incertidumbre de la simulación. Todos los resultados son DATOS SIMULADOS.

1. Relajación: Gini y entropía promedio de 40 semillas en función del tiempo. Se ajusta
   G(t) = G_eq - A1 exp(-t/tau1) - A2 exp(-t/tau2) (una relajación rápida del grueso de la
   distribución y una lenta de la cola de ricos). El tiempo de equilibrio t_eq es el primer
   momento a partir del cual la distancia promedio al valor del ensamble exacto, promediada
   en ventanas de 1000 intercambios por persona, ya no supera 2 errores estándar.
2. Fluctuaciones contra tamaño: desviación estándar del Gini entre semillas para varios N,
   comparada con la del ensamble exacto (todos los repartos igual de probables).
3. Igual probabilidad de los microestados: con 3 personas y 3 monedas se cuenta cuántas
   veces aparece cada uno de los 10 repartos posibles.

Uso:  python convergencia.py
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.optimize import curve_fit
from scipy.stats import chi2

import modelo

AQUI = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(AQUI, "figuras")
RES = os.path.join(AQUI, "resultados")

N, M0 = 2000, 20
POR_PERSONA = 20_000
MEDIR = 100                         # intercambios por persona entre mediciones
SEMILLAS = list(range(300, 340))
TAMANOS = [250, 500, 1000, 2000, 4000, 8000]
VENTANA = 10                        # mediciones (x MEDIR) para el promedio móvil del criterio


def relajacion():
    series_g, series_s = [], []
    for s in SEMILLAS:
        r = modelo.simular(N, M0, pasos=POR_PERSONA * N, cada=MEDIR * N, semilla=s)
        series_g.append(r["gini"])
        series_s.append(r["S"])
    t = r["t"]
    g = np.array(series_g)
    s_ = np.array(series_s)
    g_m, g_e = g.mean(0), g.std(0, ddof=1) / np.sqrt(len(SEMILLAS))
    s_m, s_e = s_.mean(0), s_.std(0, ddof=1) / np.sqrt(len(SEMILLAS))

    def dos_exp(x, geq, a1, tau1, a2, tau2):
        return geq - a1 * np.exp(-x / tau1) - a2 * np.exp(-x / tau2)

    sel = t >= 100
    pg, cg = curve_fit(dos_exp, t[sel], g_m[sel], p0=[0.51, 0.1, 300, 0.01, 2000],
                       sigma=g_e[sel], maxfev=20000)
    ps, cs = curve_fit(dos_exp, t[sel], s_m[sel], p0=[4.0, 0.5, 100, 0.05, 1000],
                       sigma=s_e[sel], maxfev=20000)
    # Criterio empírico (no depende de la forma del ajuste)
    g_ens = modelo.ensamble_esperado(N, M0, muestras=2000)["gini"][0]
    z = np.abs(g_m - g_ens) / np.where(g_e > 0, g_e, np.inf)   # en t=0 todas las semillas coinciden
    z_movil = np.convolve(z, np.ones(VENTANA) / VENTANA, mode="valid")
    k = next(i for i in range(len(z_movil)) if np.all(z_movil[i:] < 2))
    t_eq = t[k + VENTANA - 1]
    return t, (g_m, g_e, pg, np.sqrt(np.diag(cg))), (s_m, s_e, ps, np.sqrt(np.diag(cs))), t_eq


def fluctuaciones():
    filas = []
    for n in TAMANOS:
        gs = [modelo.gini(modelo.simular(n, M0, pasos=10_000 * n, cada=10_000 * n,
                                         semilla=500 + k)["dinero"]) for k in range(20)]
        media, desv, err = modelo.media_y_error(gs)
        ens = modelo.ensamble_esperado(n, M0, muestras=1000)["gini"]
        filas.append([n, media, err, desv, ens[0], ens[1]])
        print(f"  N={n}: G = {media:.4f} ± {err:.4f}, desv. entre semillas {desv:.4f}; "
              f"ensamble {ens[0]:.4f}, desv. {ens[1]:.4f}")
    return np.array(filas)


@njit(cache=True)
def _cuenta_microestados(pasos, cada, a, b):
    dinero = np.array([1, 1, 1])
    cuentas = np.zeros((4, 4), dtype=np.int64)
    for t in range(pasos):
        i, j = a[t], b[t]
        if i != j and dinero[i] >= 1:
            dinero[i] -= 1
            dinero[j] += 1
        if t % cada == 0:
            cuentas[dinero[0], dinero[1]] += 1
    return cuentas


def microestados():
    rng = np.random.default_rng(7)
    pasos, cada = 20_000_000, 20
    c = _cuenta_microestados(pasos, cada, rng.integers(0, 3, pasos), rng.integers(0, 3, pasos))
    estados = [(i, j, 3 - i - j) for i in range(4) for j in range(4) if i + j <= 3]
    obs = np.array([c[i, j] for i, j, _ in estados])
    esperado = obs.sum() / len(estados)
    x2 = np.sum((obs - esperado) ** 2 / esperado)
    p = chi2.sf(x2, len(estados) - 1)
    print(f"  10 microestados, {obs.sum()} observaciones: chi^2 = {x2:.1f} (9 g.l.), p = {p:.2f}")
    return estados, obs / obs.sum(), np.sqrt(obs) / obs.sum(), x2, p


def principal():
    print("Datos SIMULADOS: convergencia e incertidumbre.")
    t, (g_m, g_e, pg, eg), (s_m, s_e, ps, es), t_eq = relajacion()
    ens = modelo.ensamble_esperado(N, M0, muestras=2000)
    print(f"  Gini: G_eq = {pg[0]:.4f} ± {eg[0]:.4f}, tau1 = {pg[2]:.0f} ± {eg[2]:.0f}, "
          f"tau2 = {pg[4]:.0f} ± {eg[4]:.0f} intercambios/persona (A1={pg[1]:.3f}, A2={pg[3]:.4f}); "
          f"t_eq = {t_eq:.0f}")
    print(f"  Entropía: S_eq = {ps[0]:.4f} ± {es[0]:.4f}, tau1 = {ps[2]:.0f} ± {es[2]:.0f}, "
          f"tau2 = {ps[4]:.0f} ± {es[4]:.0f}")
    print(f"  Ensamble exacto N={N}: G = {ens['gini'][0]:.4f}, S = {ens['entropia'][0]:.4f}")
    fl = fluctuaciones()
    estados, frec, frec_e, x2, p = microestados()

    with open(os.path.join(RES, "convergencia.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cantidad", "valor", "incertidumbre"])
        w.writerow(["gini_equilibrio_ajuste", f"{pg[0]:.4f}", f"{eg[0]:.4f}"])
        w.writerow(["tau1_gini_intercambios_por_persona", f"{pg[2]:.0f}", f"{eg[2]:.0f}"])
        w.writerow(["tau2_gini_intercambios_por_persona", f"{pg[4]:.0f}", f"{eg[4]:.0f}"])
        w.writerow(["A1_gini", f"{pg[1]:.4f}", f"{eg[1]:.4f}"])
        w.writerow(["A2_gini", f"{pg[3]:.4f}", f"{eg[3]:.4f}"])
        w.writerow(["t_equilibrio_gini_criterio_2_errores_estandar", f"{t_eq:.0f}", ""])
        w.writerow(["entropia_equilibrio_ajuste", f"{ps[0]:.4f}", f"{es[0]:.4f}"])
        w.writerow(["tau1_entropia_intercambios_por_persona", f"{ps[2]:.0f}", f"{es[2]:.0f}"])
        w.writerow(["tau2_entropia_intercambios_por_persona", f"{ps[4]:.0f}", f"{es[4]:.0f}"])
        w.writerow(["gini_ensamble_exacto_N2000", f"{ens['gini'][0]:.4f}", f"{ens['gini'][1]:.4f}"])
        w.writerow(["entropia_ensamble_exacto_N2000", f"{ens['entropia'][0]:.4f}",
                    f"{ens['entropia'][1]:.4f}"])
        w.writerow(["chi2_microestados_9gl", f"{x2:.2f}", f"p={p:.3f}"])
        w.writerow([])
        w.writerow(["N", "gini_media", "error_estandar", "desv_entre_semillas",
                    "gini_ensamble", "desv_ensamble"])
        for f in fl:
            w.writerow([int(f[0])] + [f"{v:.4f}" for v in f[1:]])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    dev = np.abs(ens["gini"][0] - g_m)
    ax[0].semilogy(t, dev, ".", color="#c0504d", ms=3, label="|G ensamble − G simulado| (40 semillas)")
    ax[0].semilogy(t, pg[1] * np.exp(-t / pg[2]) + pg[3] * np.exp(-t / pg[4]), "k-",
                   label=rf"Ajuste, $\tau_1$ = {pg[2]:.0f}, $\tau_2$ = {pg[4]:.0f}")
    ax[0].semilogy(t, 2 * g_e, "--", color="0.5", label="2 errores estándar")
    ax[0].axvline(t_eq, color="#3a6ea5", ls=":", label=f"Equilibrio: t ≈ {t_eq:.0f}")
    ax[0].set(xlabel="Intercambios por persona", ylabel="Distancia al equilibrio (Gini)",
              ylim=(1e-5, 1), title=f"1. Relajación: equilibrio a partir de t ≈ {t_eq:.0f}")
    ax[0].legend(fontsize=7.5)

    ax[1].loglog(fl[:, 0], fl[:, 3], "o", color="#c0504d", label="Simulación (20 semillas)")
    ax[1].loglog(fl[:, 0], fl[:, 5], "s", mfc="none", color="k", label="Ensamble exacto")
    ref = fl[3, 5] * np.sqrt(fl[3, 0] / fl[:, 0])
    ax[1].loglog(fl[:, 0], ref, "--", color="0.5", label=r"$\propto N^{-1/2}$")
    ax[1].set_xticks(TAMANOS, [str(n) for n in TAMANOS])
    ax[1].minorticks_off()
    ax[1].set(xlabel="Número de personas N", ylabel="Desviación estándar del Gini",
              title="2. Fluctuaciones contra tamaño")
    ax[1].legend(fontsize=8)

    etiquetas = ["".join(map(str, e)) for e in estados]
    ax[2].bar(etiquetas, frec, yerr=frec_e, color="#9bb7d4", capsize=2)
    ax[2].axhline(0.1, color="k", ls="--", label="Igual probabilidad (10 %)")
    ax[2].set(xlabel="Reparto (monedas de cada persona)", ylabel="Frecuencia",
              ylim=(0, 0.13), title=f"3. Microestados (3 personas, 3 monedas): p = {p:.2f}")
    ax[2].legend(fontsize=8, loc="lower right")
    fig.suptitle("Datos simulados: convergencia, fluctuaciones e igual probabilidad")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig11_convergencia.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    principal()
