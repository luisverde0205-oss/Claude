"""
Simulación del modelo de Drăgulescu y Yakovenko (2000), "Statistical mechanics of money".

Idea: N personas con la misma cantidad de dinero hacen intercambios al azar.
En cada intercambio una persona le paga a otra una cantidad fija (dm).
Si la que paga no tiene suficiente dinero, el intercambio no ocurre (no hay deudas).
El dinero total nunca cambia, como la energía en un gas aislado.

Predicción de la física estadística: al final el dinero se reparte según
    P(m) = (1/T) * exp(-m/T),   con  T = dinero total / número de personas.

Uso:  python simulacion_dinero.py
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parámetros ----------------
N = 2000            # número de personas
M0 = 20             # dinero inicial de cada persona (todos empiezan iguales)
DM = 1              # cantidad que se paga en cada intercambio
PASOS = 4_000_000   # número total de intercambios
CADA = 20_000       # cada cuántos intercambios se guardan mediciones
SEMILLA = 42

T = M0  # "temperatura del dinero" = dinero promedio por persona


def entropia(dinero):
    """Entropía de Shannon/Gibbs de la distribución del dinero: S = -sum P ln P."""
    conteos = np.bincount(dinero)
    p = conteos[conteos > 0] / len(dinero)
    return -np.sum(p * np.log(p))


def gini(dinero):
    """Coeficiente de Gini: 0 = todos iguales, 1 = una persona lo tiene todo."""
    x = np.sort(dinero).astype(float)
    n = len(x)
    i = np.arange(1, n + 1)
    return (2 * np.sum(i * x) / (n * x.sum())) - (n + 1) / n


def simular():
    rng = np.random.default_rng(SEMILLA)
    dinero = np.full(N, M0, dtype=np.int64)
    # Se sortean de antemano todas las parejas (más rápido que sortear una por una)
    pagadores = rng.integers(0, N, PASOS)
    receptores = rng.integers(0, N, PASOS)

    tiempos, entropias, ginis = [], [], []
    for t in range(PASOS):
        i, j = pagadores[t], receptores[t]
        if i != j and dinero[i] >= DM:   # regla: nadie puede quedar con dinero negativo
            dinero[i] -= DM
            dinero[j] += DM
        if t % CADA == 0:
            tiempos.append(t)
            entropias.append(entropia(dinero))
            ginis.append(gini(dinero))

    assert dinero.sum() == N * M0, "¡El dinero total no se conservó!"
    return dinero, np.array(tiempos), np.array(entropias), np.array(ginis)


def graficar(dinero, tiempos, entropias, ginis):
    m = np.arange(0, dinero.max() + 1)
    p_sim = np.bincount(dinero, minlength=len(m)) / N
    # Versión discreta de Boltzmann (el dinero viene en monedas enteras)
    q = np.exp(-DM / T)
    p_teo = (1 - q) * q ** m

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    ax[0, 0].bar(m, p_sim, width=1, color="#9bb7d4", label="Simulación")
    ax[0, 0].plot(m, p_teo, "k-", lw=2, label=r"Boltzmann: $e^{-m/T}/T$")
    ax[0, 0].axvline(M0, color="r", ls="--", label=f"Dinero inicial ({M0})")
    ax[0, 0].set(xlabel="Dinero m", ylabel="Fracción de personas P(m)",
                 title="1. Distribución final del dinero")
    ax[0, 0].legend()

    ax[0, 1].semilogy(m, p_sim, "o", ms=4, color="#3a6ea5", label="Simulación")
    ax[0, 1].semilogy(m, p_teo, "k-", lw=2, label="Boltzmann (recta)")
    ax[0, 1].set(xlabel="Dinero m", ylabel="P(m) (escala log)",
                 title="2. Escala logarítmica: la exponencial se ve recta")
    ax[0, 1].legend()

    s_max = -np.sum(p_teo[p_teo > 0] * np.log(p_teo[p_teo > 0]))
    ax[1, 0].plot(tiempos / N, entropias, color="#3a6ea5")
    ax[1, 0].axhline(s_max, color="k", ls="--", label="Máximo teórico")
    ax[1, 0].set(xlabel="Intercambios por persona", ylabel="Entropía S",
                 title="3. La entropía crece hasta su máximo")
    ax[1, 0].legend()

    ax[1, 1].plot(tiempos / N, ginis, color="#c0504d")
    ax[1, 1].axhline(0.5, color="k", ls="--", label="Teoría: Gini = 0.5")
    ax[1, 1].set(xlabel="Intercambios por persona", ylabel="Gini",
                 title="4. La desigualdad aparece sola")
    ax[1, 1].legend()

    fig.suptitle(f"Modelo de Drăgulescu–Yakovenko: N={N} personas, "
                 f"dinero inicial={M0}, pago por intercambio={DM}")
    fig.tight_layout()
    fig.savefig("figuras/simulacion_dinero.png", dpi=130)
    return s_max


if __name__ == "__main__":
    dinero, tiempos, entropias, ginis = simular()
    s_max = graficar(dinero, tiempos, entropias, ginis)
    print(f"Dinero total conservado: {dinero.sum()} (= {N} x {M0})")
    print(f"Temperatura del dinero T = {T}")
    print(f"Fracción con menos que el promedio: {np.mean(dinero < T):.3f} "
          f"(teoría ≈ {1 - np.exp(-1):.3f})")
    print(f"Fracción sin dinero (m=0): {np.mean(dinero == 0):.3f} "
          f"(teoría ≈ {1 - np.exp(-DM / T):.3f})")
    print(f"Gini final: {ginis[-1]:.3f} (teoría 0.5)")
    print(f"Entropía final: {entropias[-1]:.3f} (máximo teórico {s_max:.3f})")
    print("Figura guardada en figuras/simulacion_dinero.png")
