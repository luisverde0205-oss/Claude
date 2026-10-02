"""
Compara la predicción del modelo (distribución exponencial, G = 1/2) con los países
más desiguales del mundo según el Banco Mundial (Poverty and Inequality Platform).

Datos REALES: índice de Gini y participación en el ingreso/consumo por grupos,
dato más reciente de cada país. Se guardan en resultados/banco_mundial_crudo.csv.

Uso:  python comparar_paises.py            # descarga del API del Banco Mundial
      python comparar_paises.py --local    # usa el CSV ya guardado
"""
import csv
import json
import os
import sys
import urllib.request

import matplotlib.pyplot as plt
import numpy as np

import modelo

AQUI = os.path.dirname(os.path.abspath(__file__))
CRUDO = os.path.join(AQUI, "resultados", "banco_mundial_crudo.csv")

# Indicador -> nombre corto. Participaciones en % del ingreso (o consumo) total.
INDICADORES = {
    "SI.POV.GINI": "gini",
    "SI.DST.FRST.10": "d1",    # 10 % más pobre
    "SI.DST.FRST.20": "q1",    # 20 % más pobre
    "SI.DST.02ND.20": "q2",
    "SI.DST.03RD.20": "q3",
    "SI.DST.04TH.20": "q4",
    "SI.DST.05TH.20": "q5",    # 20 % más rico
    "SI.DST.10TH.10": "d10",   # 10 % más rico
}
N_DESIGUALES = 10
REFERENCIAS = ["MEX"]          # se agregan aunque no estén en el top


def descargar():
    datos = {}
    for ind, corto in INDICADORES.items():
        url = (f"https://api.worldbank.org/v2/country/all/indicator/{ind}"
               "?format=json&per_page=400&mrnev=1")
        with urllib.request.urlopen(url, timeout=60) as r:
            filas = json.load(r)[1]
        for f in filas:
            if f["value"] is None:
                continue
            iso = f["countryiso3code"]
            d = datos.setdefault(iso, {"iso3": iso, "pais": f["country"]["value"]})
            d[corto] = f["value"]
            d[f"anio_{corto}"] = f["date"]
    columnas = ["iso3", "pais"] + [c for k in INDICADORES.values() for c in (k, f"anio_{k}")]
    with open(CRUDO, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas)
        w.writeheader()
        for d in datos.values():
            w.writerow({c: d.get(c, "") for c in columnas})


def leer():
    with open(CRUDO) as fh:
        filas = list(csv.DictReader(fh))
    out = []
    for f in filas:
        if not f["gini"] or not f["d10"]:
            continue
        # Solo se usan países cuyas participaciones son del mismo año que el Gini
        if any(f[f"anio_{k}"] != f["anio_gini"] for k in INDICADORES.values()):
            continue
        out.append({k: (float(v) if k in INDICADORES.values() else v) for k, v in f.items()})
    return out


def lorenz_pais(f):
    """Puntos de la curva de Lorenz a partir de deciles y quintiles (en fracciones)."""
    p = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    acum = np.cumsum([f["q1"], f["q2"], f["q3"], f["q4"]]) / 100
    L = [0, f["d1"] / 100, acum[0], acum[1], acum[2], acum[3], 1 - f["d10"] / 100, 1.0]
    return np.array(p), np.array(L)


def principal():
    if "--local" not in sys.argv:
        descargar()
    paises = leer()
    paises.sort(key=lambda f: -f["gini"])
    sel = paises[:N_DESIGUALES] + [f for f in paises if f["iso3"] in REFERENCIAS]

    # Predicciones del modelo (exponencial)
    Lexp = modelo.lorenz_exponencial
    mod = {"gini": 50.0, "d1": 100 * Lexp(0.1), "q1": 100 * Lexp(0.2),
           "mitad": 100 * Lexp(0.5), "q5": 100 * (1 - Lexp(0.8)), "d10": 100 * (1 - Lexp(0.9))}

    filas = []
    print(f"{'País':28s}{'Año':>6s}{'Gini':>7s}{'10% pobre':>11s}{'20% pobre':>11s}"
          f"{'20% rico':>10s}{'10% rico':>10s}{'Rico/pobre':>12s}")
    print(f"{'MODELO (exponencial)':28s}{'':>6s}{mod['gini']:7.1f}{mod['d1']:11.1f}"
          f"{mod['q1']:11.1f}{mod['q5']:10.1f}{mod['d10']:10.1f}{mod['d10']/mod['d1']:12.0f}")
    for f in sel:
        razon = f["d10"] / f["d1"]
        print(f"{f['pais'][:27]:28s}{f['anio_gini']:>6s}{f['gini']:7.1f}{f['d1']:11.1f}"
              f"{f['q1']:11.1f}{f['q5']:10.1f}{f['d10']:10.1f}{razon:12.0f}")
        filas.append([f["pais"], f["iso3"], f["anio_gini"], f["gini"], f["d1"], f["q1"],
                      f["q5"], f["d10"], round(razon, 1)])
    with open(os.path.join(AQUI, "resultados", "comparacion_paises.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pais", "iso3", "anio", "gini", "pct_10_pobre", "pct_20_pobre",
                    "pct_20_rico", "pct_10_rico", "razon_10rico_10pobre"])
        w.writerow(["MODELO exponencial", "-", "-", 50.0, round(mod["d1"], 2), round(mod["q1"], 2),
                    round(mod["q5"], 2), round(mod["d10"], 2), round(mod["d10"] / mod["d1"], 1)])
        w.writerows(filas)

    graficar(sel, mod)
    graficar_gini(sel)


def graficar_gini(sel):
    """Barras del índice de Gini de cada país comparado con el del modelo (50)."""
    sel = sorted(sel, key=lambda f: f["gini"])
    es = {"South Africa": "Sudáfrica", "Brazil": "Brasil", "Zimbabwe": "Zimbabue",
          "Panama": "Panamá", "Mexico": "México"}
    nombres = [f"{es.get(f['pais'], f['pais'])} ({f['anio_gini']})" for f in sel]
    g = [f["gini"] for f in sel]
    colores = ["#c0504d" if v > 50 else "#3a6ea5" for v in g]
    fig, ax = plt.subplots(figsize=(8, 5))
    barras = ax.barh(nombres, g, color=colores)
    ax.bar_label(barras, fmt="%.1f", padding=3, fontsize=9)
    ax.axvline(50, color="k", ls="--", lw=1.5)
    ax.text(50.3, -0.9, "Modelo de azar puro = 50", fontsize=9)
    ax.set(xlabel="Índice de Gini (datos reales, Banco Mundial)", xlim=(35, 63),
           ylim=(-1.3, len(sel) - 0.5),
           title="Rojo: más desigual que el azar puro · Azul: menos desigual")
    fig.tight_layout()
    fig.savefig(os.path.join(AQUI, "figuras", "fig7_gini_paises.png"), dpi=150)
    plt.close(fig)


def graficar(sel, mod):
    top = sel[:4] + [f for f in sel if f["iso3"] in REFERENCIAS]
    colores = ["#c0504d", "#d98b3a", "#8e6bb5", "#6a9a4f", "#3a6ea5"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

    p = np.linspace(0, 1, 400)
    ax[0].plot([0, 1], [0, 1], color="0.6", ls=":", label="Igualdad perfecta")
    ax[0].plot(p, modelo.lorenz_exponencial(p), "k-", lw=2.5, label="Modelo (G = 0.50)")
    for f, c in zip(top, colores):
        pp, L = lorenz_pais(f)
        ax[0].plot(pp, L, "o-", color=c, ms=4, lw=1.3,
                   label=f"{f['pais']} {f['anio_gini']} (G = {f['gini']/100:.2f})")
    ax[0].set(xlabel="Fracción de personas (de más pobre a más rica)",
              ylabel="Fracción del ingreso total", aspect="equal",
              title="Curvas de Lorenz: modelo vs. países")
    ax[0].legend(fontsize=7.5, loc="upper left")

    # Diferencia de participación respecto al modelo
    grupos = ["d1", "q1", "q5", "d10"]
    etiquetas = ["10 % más\npobre", "20 % más\npobre", "20 % más\nrico", "10 % más\nrico"]
    x = np.arange(len(grupos))
    ancho = 0.8 / len(top)
    for k, (f, c) in enumerate(zip(top, colores)):
        dif = [f[g] - mod[g] for g in grupos]
        ax[1].bar(x + (k - len(top) / 2 + 0.5) * ancho, dif, ancho, color=c, label=f["pais"])
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xticks(x, etiquetas)
    ax[1].set(ylabel="Puntos porcentuales respecto al modelo",
              title="¿Quién recibe más (+) o menos (−) que en el modelo?")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(AQUI, "figuras", "fig6_paises.png"), dpi=150)


if __name__ == "__main__":
    principal()
