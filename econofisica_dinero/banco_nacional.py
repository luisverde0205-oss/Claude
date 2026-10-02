"""
Extensión del modelo: un "banco nacional" inyecta dinero nuevo con monedas MARCADAS
(trazadoras), para seguir su recorrido. Todos los resultados son DATOS SIMULADOS.

Cada persona tiene monedas normales y monedas marcadas. Al pagar, la moneda que
entrega se escoge al azar de su bolsa, así que es marcada con probabilidad
(marcadas / total). Así las marcadas circulan exactamente igual que las demás.

Canales de entrada del dinero nuevo:
  1. Universal: la misma cantidad a todas las personas.
  2. Apoyos:    la misma cantidad solo al 50 % más pobre.
  3. Financiero: la misma cantidad solo al 10 % más rico (estilizado: el dinero
                entra por quienes ya tienen activos o acceso al crédito).

Experimento A (pulso): una sola inyección del 10 % del dinero total y se sigue
qué grupo tiene las monedas marcadas, comparado con la parte de dinero que tiene.

Experimento B (periódico): cada 100 intercambios por persona se inyecta el 5 % del
dinero inicial (20 inyecciones; al final el dinero se duplica), y se mide el Gini.

Uso:  python banco_nacional.py
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

import modelo

AQUI = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(AQUI, "figuras")
RES = os.path.join(AQUI, "resultados")

N, M0 = 1000, 20
EQUILIBRIO = 1000            # intercambios por persona antes de inyectar
PULSO = 0.10                 # fracción del dinero total en el pulso
DESPUES = 3000               # intercambios por persona después del pulso
MEDIR_CADA = 20              # intercambios por persona entre mediciones
PERIODO = 100                # experimento B: intercambios por persona entre inyecciones
TASA = 0.05                  # experimento B: fracción del dinero inicial por inyección
                             # (1 moneda a cada persona en el canal universal)
DURACION_B = 2000            # experimento B: intercambios por persona
SEMILLAS = [21, 22, 23]

CANALES = {"universal": "1. Universal (a todos)",
           "apoyos": "2. Apoyos (50 % más pobre)",
           "financiero": "3. Financiero (10 % más rico)"}
COLORES = {"control": "0.4", "universal": "#3a6ea5", "apoyos": "#6a9a4f", "financiero": "#c0504d"}
GRUPOS = [("50 % más pobre", 0.0, 0.5), ("40 % medio", 0.5, 0.9), ("10 % más rico", 0.9, 1.0)]


def intercambiar(libres, marcadas, pasos, rng):
    a = rng.integers(0, N, pasos)
    b = rng.integers(0, N, pasos)
    u = rng.random(pasos)
    for i, j, x in zip(a, b, u):
        if i == j:
            continue
        tot = libres[i] + marcadas[i]
        if tot < 1:
            continue
        if x * tot < marcadas[i]:
            marcadas[i] -= 1
            marcadas[j] += 1
        else:
            libres[i] -= 1
            libres[j] += 1


def inyectar(libres, marcadas, monedas, canal, rng):
    """Reparte `monedas` marcadas por partes iguales entre los receptores del canal.

    Si no se dividen exacto, las sobrantes se dan a receptores escogidos al azar."""
    total = libres + marcadas
    orden = np.argsort(total, kind="stable")
    if canal == "universal":
        receptores = orden
    elif canal == "apoyos":
        receptores = orden[: N // 2]
    else:
        receptores = orden[int(0.9 * N):]
    base, resto = divmod(monedas, len(receptores))
    marcadas[receptores] += base
    marcadas[rng.choice(receptores, resto, replace=False)] += 1


def enriquecimiento(libres, marcadas):
    """Para cada grupo: (parte de las monedas marcadas) / (parte de todo el dinero)."""
    total = libres + marcadas
    orden = np.argsort(total, kind="stable")
    out = []
    for _, lo, hi in GRUPOS:
        idx = orden[int(lo * N):int(hi * N)]
        out.append((marcadas[idx].sum() / marcadas.sum()) / (total[idx].sum() / total.sum()))
    return out


def equilibrar(semilla):
    rng = np.random.default_rng(semilla)
    libres = np.full(N, M0, dtype=np.int64)
    marcadas = np.zeros(N, dtype=np.int64)
    intercambiar(libres, marcadas, EQUILIBRIO * N, rng)
    return libres, marcadas, rng


def experimento_pulso():
    tiempos = np.arange(0, DESPUES + 1, MEDIR_CADA)
    res = {c: {"enr": [], "gini": []} for c in CANALES}
    for s in SEMILLAS:
        libres0, marcadas0, _ = equilibrar(s)
        for c in CANALES:
            rng = np.random.default_rng(1000 + s)
            libres, marcadas = libres0.copy(), marcadas0.copy()
            g_antes = modelo.gini(libres + marcadas)
            inyectar(libres, marcadas, int(PULSO * N * M0), c, rng)
            enr, gin = [enriquecimiento(libres, marcadas)], [modelo.gini(libres + marcadas)]
            for _ in tiempos[1:]:
                intercambiar(libres, marcadas, MEDIR_CADA * N, rng)
                enr.append(enriquecimiento(libres, marcadas))
                gin.append(modelo.gini(libres + marcadas))
            assert (libres + marcadas).sum() == N * M0 * (1 + PULSO)
            res[c]["enr"].append(np.array(enr))
            res[c]["gini"].append(np.array([g_antes] + gin))
    return tiempos, res


def tiempo_mezcla(t, enr, tolerancia=0.15):
    """Primer tiempo en que todos los grupos quedan a menos de 15 % de la mezcla perfecta (=1)."""
    dentro = np.all(np.abs(enr - 1) < tolerancia, axis=1)
    return int(t[np.argmax(dentro)]) if dentro.any() else np.nan


def experimento_periodico():
    n_iny = DURACION_B // PERIODO
    res = {c: [] for c in ["control"] + list(CANALES)}
    for s in SEMILLAS:
        libres0, marcadas0, _ = equilibrar(s)
        for c in res:
            rng = np.random.default_rng(2000 + s)
            libres, marcadas = libres0.copy(), marcadas0.copy()
            serie = [modelo.gini(libres + marcadas)]
            for _ in range(n_iny):
                if c != "control":
                    inyectar(libres, marcadas, int(TASA * N * M0), c, rng)
                intercambiar(libres, marcadas, PERIODO * N, rng)
                serie.append(modelo.gini(libres + marcadas))
            res[c].append(np.array(serie))
    return np.arange(n_iny + 1) * PERIODO, res


def principal():
    print("Datos SIMULADOS: inyección de dinero marcado por tres canales.")
    t, pulso = experimento_pulso()
    t_b, per = experimento_periodico()

    filas = []
    for c, nombre in CANALES.items():
        enr = np.mean(pulso[c]["enr"], axis=0)
        gin = np.mean(pulso[c]["gini"], axis=0)
        mezcla = tiempo_mezcla(t, enr)
        g_fin = np.mean([g[-1] for g in per[c]])
        g_des = np.std([g[-1] for g in per[c]])
        filas.append([nombre, *[round(x, 2) for x in enr[0]], round(gin[0], 3), round(gin[1], 3),
                      round(gin[-1], 3), mezcla, round(g_fin, 3), round(g_des, 3)])
        print(f"  {nombre}: enriquecimiento inicial (pobre, medio, rico) = "
              f"{', '.join(f'{x:.2f}' for x in enr[0])}; Gini {gin[0]:.3f} -> {gin[1]:.3f} "
              f"-> {gin[-1]:.3f}; mezcla en ~{mezcla} intercambios/persona; "
              f"Gini con inyección periódica {g_fin:.3f} ± {g_des:.3f}")
    g_ctrl = [g[-1] for g in per["control"]]
    print(f"  Control (sin inyección): Gini {np.mean(g_ctrl):.3f} ± {np.std(g_ctrl):.3f}")
    with open(os.path.join(RES, "banco_nacional.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["canal", "enr_inicial_50_pobre", "enr_inicial_40_medio", "enr_inicial_10_rico",
                    "gini_antes", "gini_justo_despues", "gini_final_pulso",
                    "tiempo_mezcla_intercambios_por_persona",
                    "gini_inyeccion_periodica", "gini_inyeccion_periodica_desv"])
        w.writerows(filas)
        w.writerow(["Control (sin inyección)", "", "", "", "", "", "", "",
                    round(np.mean(g_ctrl), 3), round(np.std(g_ctrl), 3)])

    graficar(t, pulso, t_b, per)


def graficar(t, pulso, t_b, per):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    estilos = ["-", "--", ":"]
    for k, (c, nombre) in enumerate(CANALES.items()):
        enr = np.mean(pulso[c]["enr"], axis=0)
        for g, (gnom, _, _) in enumerate(GRUPOS):
            ax[k].plot(t, enr[:, g], estilos[g], color=COLORES[c], lw=2, label=gnom)
        ax[k].axhline(1, color="k", lw=0.8)
        ax[k].set(xlim=(0, 400), xlabel="Intercambios por persona después de la inyección",
                  title=nombre)
        ax[k].legend(fontsize=8)
    ax[0].set_ylabel("Monedas nuevas que tiene el grupo\n÷ dinero total que tiene el grupo")
    fig.suptitle("Datos simulados: recorrido de las monedas marcadas (1 = mezcla perfecta)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig9_rastreo_monedas.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    tg = np.concatenate([[-100], t])
    for c, nombre in CANALES.items():
        ax[0].plot(tg, np.mean(pulso[c]["gini"], axis=0), color=COLORES[c], lw=2, label=nombre)
    ax[0].axhline(0.5, color="k", ls=":", lw=1)
    ax[0].axvline(0, color="k", lw=0.8, ls="--")
    ax[0].set(xlim=(-100, DESPUES), xlabel="Intercambios por persona (la inyección ocurre en 0)",
              ylabel="Coeficiente de Gini", title="A. Una sola inyección (10 % del dinero)")
    ax[0].legend(fontsize=8)
    for c in per:
        nombre = CANALES.get(c, "Control (sin inyección)")
        ax[1].plot(t_b, np.mean(per[c], axis=0), color=COLORES[c], lw=2, label=nombre)
    ax[1].set(xlabel="Intercambios por persona", ylabel="Coeficiente de Gini",
              title="B. Inyección periódica (5 % cada 100 intercambios)")
    ax[1].legend(fontsize=8)
    fig.suptitle("Datos simulados: efecto de la inyección sobre la desigualdad")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig10_inyeccion_gini.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    principal()
