"""
Extensión del modelo: crecimiento anual del dinero tomando como referencia el S&P 500
(rendimiento promedio histórico de aproximadamente 10 % anual).

Todos los resultados son DATOS SIMULADOS.

Cómo se define "un año": cada persona hace en promedio INTERCAMBIOS_POR_ANIO
intercambios (unos 50, casi uno por semana). Al terminar cada año se aplica
el crecimiento según el escenario:

  0. Sin crecimiento     (control: el modelo original)
  A. Todos +10 %         (lo que pidió el autor: a TODOS se les sube 10 % su dinero)
  B. Solo invierte quien tiene más que el promedio (+10 % solo para ellos)
  C. Todos invierten, pero cada persona obtiene un rendimiento distinto al azar
     (promedio 10 %, desviación 20 %), como si cada quien escogiera acciones distintas

Para que el modelo funcione cuando el dinero crece, el pago en cada intercambio
es una fracción fija (1/10) del dinero promedio del momento, como precios que
suben junto con la economía.

Uso:  python crecimiento.py
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

import modelo

AQUI = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(AQUI, "figuras")
RES = os.path.join(AQUI, "resultados")

N = 1000
M0 = 100.0
FRACCION_PAGO = 0.1          # pago = 10 % del dinero promedio
INTERCAMBIOS_POR_ANIO = 50   # por persona (movilidad alta, caso principal)
MOVILIDADES = [50, 10, 2]    # intercambios por persona al año: alta, media, baja
CALENTAMIENTO = 500          # intercambios por persona sin crecimiento, para llegar al equilibrio
ANIOS = 30                   # años con crecimiento
RENDIMIENTO = 0.10           # referencia S&P 500
VOLATILIDAD = 0.20           # solo escenario C
SEMILLAS = [1, 2, 3]

ESCENARIOS = {
    "0": "Sin crecimiento (control)",
    "A": "A. Todos +10 % al año",
    "B": "B. Solo quien tiene más que el promedio +10 %",
    "C": "C. Rendimiento al azar (promedio 10 %, desv. 20 %)",
}
ESTILO = {"0": dict(ls="--", zorder=5)}   # el control coincide con A; se dibuja punteado encima
COLORES = {"0": "0.4", "A": "#3a6ea5", "B": "#c0504d", "C": "#d98b3a"}


def intercambiar(dinero, pasos, rng):
    """Intercambios del modelo original con pago = FRACCION_PAGO x promedio actual."""
    dm = FRACCION_PAGO * dinero.mean()   # el promedio no cambia dentro del año
    a = rng.integers(0, N, pasos)
    b = rng.integers(0, N, pasos)
    for i, j in zip(a, b):
        if i != j and dinero[i] >= dm:
            dinero[i] -= dm
            dinero[j] += dm


def crecer(dinero, escenario, rng):
    if escenario == "A":
        dinero *= 1 + RENDIMIENTO
    elif escenario == "B":
        ricos = dinero > dinero.mean()
        dinero[ricos] *= 1 + RENDIMIENTO
    elif escenario == "C":
        r = rng.normal(RENDIMIENTO, VOLATILIDAD, N)
        dinero *= np.maximum(1 + r, 0.0)   # no se puede perder más del 100 %


def medir(dinero):
    x = np.sort(dinero)
    total = x.sum()
    return {
        "promedio": x.mean(),
        "gini": modelo.gini(x),
        "pct_10_rico": 100 * x[int(0.9 * N):].sum() / total,
        "pct_50_pobre": 100 * x[: N // 2].sum() / total,
        "prom_10_rico": x[int(0.9 * N):].mean(),
        "prom_50_pobre": x[: N // 2].mean(),
        "maximo": x[-1],
    }


def correr(escenario, semilla, por_anio=INTERCAMBIOS_POR_ANIO):
    rng = np.random.default_rng(semilla)
    # Casi iguales (±5 %) para que el dinero no quede atrapado en múltiplos exactos del pago,
    # lo que daría un Gini artificialmente distinto entre escenarios.
    dinero = M0 * (1 + 0.05 * rng.uniform(-1, 1, N))
    dinero *= M0 / dinero.mean()
    intercambiar(dinero, CALENTAMIENTO * N, rng)
    serie = [medir(dinero)]
    for _ in range(ANIOS):
        intercambiar(dinero, por_anio * N, rng)
        crecer(dinero, escenario, rng)
        serie.append(medir(dinero))
    return serie, dinero


def graficar(series, finales):
    anios = np.arange(ANIOS + 1)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for e, nombre in ESCENARIOS.items():
        s = series[e]
        c = COLORES[e]
        prom = np.mean([[f["promedio"] for f in s_] for s_ in s], axis=0)
        gin = np.array([[f["gini"] for f in s_] for s_ in s])
        top = np.mean([[f["pct_10_rico"] for f in s_] for s_ in s], axis=0)
        brecha = np.mean([[f["prom_10_rico"] - f["prom_50_pobre"] for f in s_] for s_ in s], axis=0)
        ax[0, 0].plot(anios, prom / M0, color=c, lw=2, label=nombre, **ESTILO.get(e, {}))
        ax[0, 1].plot(anios, gin.mean(0), color=c, lw=2, label=nombre, **ESTILO.get(e, {}))
        ax[0, 1].fill_between(anios, gin.min(0), gin.max(0), color=c, alpha=0.15)
        ax[1, 0].plot(anios, top, color=c, lw=2, label=nombre, **ESTILO.get(e, {}))
        ax[1, 1].plot(anios, brecha / M0, color=c, lw=2, label=nombre, **ESTILO.get(e, {}))
    ax[0, 0].set(yscale="log", xlabel="Año", ylabel="Dinero promedio (veces el inicial)",
                 title="1. ¿Cuánto crece la economía?")
    ax[0, 1].set(xlabel="Año", ylabel="Coeficiente de Gini", title="2. Desigualdad relativa (Gini)")
    ax[1, 0].set(xlabel="Año", ylabel="% del dinero total", title="3. Parte del 10 % más rico")
    ax[1, 1].set(yscale="log", xlabel="Año",
                 ylabel="Brecha (veces el dinero inicial)",
                 title="4. Brecha absoluta: prom. 10 % rico − prom. 50 % pobre")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Datos simulados: crecimiento anual tipo S&P 500 (N = 1000, 3 semillas)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig8_crecimiento.png"), dpi=150)
    plt.close(fig)

    # Forma de la distribución final: fracción de personas con más de x veces el promedio
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for e, nombre in ESCENARIOS.items():
        x = np.sort(np.concatenate([d / d.mean() for d in finales[e]]))
        cola = 1 - np.arange(len(x)) / len(x)
        ax.loglog(x, cola, color=COLORES[e], lw=2, label=nombre, **ESTILO.get(e, {}))
    xs = np.linspace(0.01, 12, 200)
    ax.loglog(xs, np.exp(-xs), "k:", label=r"Exponencial $e^{-x}$")
    ax.set(xlim=(0.1, 20), ylim=(1 / 4000, 1.1),
           xlabel="x = dinero / dinero promedio",
           ylabel="Fracción de personas con más de x")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig9_cola_ricos.png"), dpi=150)
    plt.close(fig)


def principal():
    print("Datos SIMULADOS. Crecimiento anual de referencia: S&P 500 ≈ 10 %.")
    series, finales, filas = {}, {}, []
    for e, nombre in ESCENARIOS.items():
        series[e], finales[e] = [], []
        for s in SEMILLAS:
            serie, dinero = correr(e, s)
            series[e].append(serie)
            finales[e].append(dinero)
        ini = series[e][0][0]
        fin = {k: np.mean([sr[-1][k] for sr in series[e]]) for k in ini}
        gdes = np.std([sr[-1]["gini"] for sr in series[e]])
        g0 = np.mean([sr[0]["gini"] for sr in series[e]])
        brecha0 = np.mean([sr[0]["prom_10_rico"] - sr[0]["prom_50_pobre"] for sr in series[e]])
        brecha = fin["prom_10_rico"] - fin["prom_50_pobre"]
        fila = [nombre, round(fin["promedio"] / M0, 2), round(g0, 3), round(fin["gini"], 3),
                round(gdes, 3), round(fin["pct_10_rico"], 1), round(fin["pct_50_pobre"], 1),
                round(brecha / brecha0, 1), round(fin["maximo"] / fin["promedio"], 1)]
        filas.append(fila)
        print(f"  {nombre}: promedio x{fila[1]}, Gini {fila[2]} -> {fila[3]} ± {fila[4]}, "
              f"10% rico {fila[5]} %, 50% pobre {fila[6]} %, brecha x{fila[7]}, "
              f"más rico = {fila[8]} veces el promedio")
    with open(os.path.join(RES, "crecimiento.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["escenario", "promedio_final_vs_inicial", "gini_anio0", "gini_anio30",
                    "gini_desv", "pct_10_rico", "pct_50_pobre", "brecha_absoluta_crece_x",
                    "mas_rico_vs_promedio"])
        w.writerows(filas)
    graficar(series, finales)
    movilidad()


def movilidad():
    """Gini y parte del 10 % más rico en el año 30 según cuántos intercambios hay al año."""
    print("Sensibilidad a la movilidad (intercambios por persona al año):")
    filas = []
    for k in MOVILIDADES:
        for e in ESCENARIOS:
            fin = [correr(e, s, k)[0][-1] for s in SEMILLAS]
            g = [f["gini"] for f in fin]
            filas.append([k, ESCENARIOS[e], round(np.mean(g), 3), round(np.std(g), 3),
                          round(np.mean([f["pct_10_rico"] for f in fin]), 1),
                          round(np.mean([f["pct_50_pobre"] for f in fin]), 1)])
            print(f"  {k:3d}/año  {ESCENARIOS[e]}: Gini {filas[-1][2]} ± {filas[-1][3]}, "
                  f"10% rico {filas[-1][4]} %")
    with open(os.path.join(RES, "crecimiento_movilidad.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["intercambios_por_anio", "escenario", "gini", "gini_desv",
                    "pct_10_rico", "pct_50_pobre"])
        w.writerows(filas)

    # Figura: Gini del año 30 contra países reales de referencia
    ref = {}
    with open(os.path.join(RES, "comparacion_paises.csv")) as fh:
        for f in csv.DictReader(fh):
            if f["iso3"] in ("NAM", "COL", "MEX"):
                ref[f["pais"]] = float(f["gini"]) / 100
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ancho = 0.2
    for k, e in enumerate(ESCENARIOS):
        vals = [r[2] for r in filas if r[1] == ESCENARIOS[e]]
        errs = [r[3] for r in filas if r[1] == ESCENARIOS[e]]
        x = np.arange(len(MOVILIDADES)) + (k - 1.5) * ancho
        ax.bar(x, vals, ancho, yerr=errs, color=COLORES[e], label=ESCENARIOS[e], capsize=2)
    estilos = {"Namibia": "-", "Colombia": "--", "Mexico": ":"}
    for pais, g in ref.items():
        ax.axhline(g, color="k", lw=1, ls=estilos.get(pais, "-"))
        ax.text(len(MOVILIDADES) - 0.45, g + 0.004, f"{pais} (real): {g:.2f}", fontsize=8)
    ax.set_xticks(range(len(MOVILIDADES)),
                  [f"Alta\n({m} intercambios/año)" if i == 0 else
                   f"{'Media' if i == 1 else 'Baja'}\n({m} intercambios/año)"
                   for i, m in enumerate(MOVILIDADES)])
    ax.set(ylabel="Gini en el año 30", ylim=(0.35, 0.72), xlim=(-0.5, len(MOVILIDADES) + 0.4),
           title="Datos simulados: Gini tras 30 años según la movilidad")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig10_movilidad_gini.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    principal()
