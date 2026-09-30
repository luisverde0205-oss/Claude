"""Paso 6. Figuras a partir de resultados/ (no recalcula nada)."""
import json
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import comun as c

INK, INK2, GR, GRID = '#0b0b0b', '#52514e', '#9c9b96', '#ecebe6'
COL = {'China 2007': '#2a78d6', 'EE.UU. 2008': '#eb6834', 'India 2019': '#1baf7a', 'Rusia 2021': '#4a3aa7'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.edgecolor': GR, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 11, 'axes.titlelocation': 'left'})
c.FIG.mkdir(exist_ok=True)


def estilo(ax):
    ax.grid(color=GRID, lw=.7); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)


def guardar(fig, nombre):
    fig.tight_layout(); fig.savefig(c.FIG / nombre, dpi=190, facecolor='white'); plt.close(fig)


# 1. Diagrama de Gabbard (leyes de Kepler)
k = pd.read_csv(c.RES / 'kepler_fengyun1c.csv')
fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.scatter(k.T_min, k.h_apogeo_km, s=4, color='#2a78d6', alpha=.6, lw=0, label='Apogeo, a(1+e) − R$_T$')
ax.scatter(k.T_min, k.h_perigeo_km, s=4, color='#eb6834', alpha=.6, lw=0, label='Perigeo, a(1−e) − R$_T$')
ax.axhline(865, color=GR, lw=1, ls=':'); ax.set_ylim(250, 2800); ax.text(k.T_min.max(), 900, 'altura de la intercepción (~865 km)', ha='right', fontsize=8, color=INK2)
ax.set_xlabel('Periodo T (min), tercera ley: T = 2π√(a³/μ)'); ax.set_ylabel('Altura (km)')
ax.set_title(f'Diagrama de Gabbard: {len(k)} fragmentos de Fengyun-1C (sep. 2026)')
ax.legend(frameon=False, fontsize=8.5, markerscale=3, loc='upper left'); estilo(ax)
guardar(fig, 'fig1_gabbard_kepler.png')

# 2. Supervivencia observada
f = pd.read_csv(c.RES / 'fragmentos_pruebas.csv'); sup = pd.read_csv(c.RES / 'supervivencia_resumen.csv')
fig, ax = plt.subplots(figsize=(7.5, 5.2))
tt = np.logspace(-2.5, np.log10(21), 500)
for ev, g in f.groupby('evento'):
    m = tt <= g.vida_anios.max() + 1e-9
    frac = np.array([(g.vida_anios > x).mean() for x in tt[m]]) * 100
    r = sup[sup.evento == ev].iloc[0]
    ax.plot(tt[m], frac, color=COL[ev], lw=2, label=f'{ev}: {r.altitud_km} km, {r.fragmentos} fragmentos')
ax.set_xscale('log'); ax.set_xlim(3e-3, 60); ax.set_ylim(-3, 103)
ax.set_xticks([0.01, 0.1, 1, 10]); ax.set_xticklabels(['4 días', '1 mes', '1 año', '10 años'])
ax.set_xlabel('Tiempo desde la prueba'); ax.set_ylabel('Fragmentos catalogados en órbita (%)')
ax.set_title('Supervivencia observada de los fragmentos (SATCAT)')
ax.legend(frameon=False, fontsize=8.3, loc='lower right', bbox_to_anchor=(1.0, 0.2)); estilo(ax)
guardar(fig, 'fig2_supervivencia.png')

# 3. Control con dn/dt: B* no es un coeficiente físico
d = pd.read_csv(c.RES / 'decaimiento_fengyun1c.csv')
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
a1.scatter(d.h_perigeo_km, d.razon_modelo_obs, s=5, color='#2a78d6', alpha=.5, lw=0)
a1.set_yscale('log'); a1.set_xlabel('Altura del perigeo (km)'); a1.set_ylabel('dn/dt modelo ÷ dn/dt observado')
a1.set_title('A. Con C$_D$A/m = 12.74·B* el modelo cae demasiado lento'); a1.axhline(1, color=INK, lw=1.2)
a1.text(d.h_perigeo_km.min(), 1.15, 'acuerdo perfecto', fontsize=8, color=INK2); estilo(a1)
a2.hist(np.log10(d.CdA_m_calibrado), bins=40, color='#1baf7a', edgecolor='white')
a2.set_xlabel('log$_{10}$(C$_D$A/m calibrado, m²/kg)'); a2.set_ylabel('Fragmentos')
med = np.median(d.CdA_m_calibrado); a2.axvline(np.log10(med), color=INK, lw=1.5)
a2.text(np.log10(med) + 0.05, a2.get_ylim()[1] * 0.9, f'mediana {med:.2f} m²/kg', fontsize=8.5, color=INK)
a2.set_title('B. Coeficiente de frenado calibrado con dn/dt observado'); estilo(a2)
guardar(fig, 'fig3_calibracion_frenado.png')

# 4. Pasado observado y futuro modelado de los fragmentos chinos
ch = f[f.evento == 'China 2007'].copy(); ch['DECAY_DATE'] = pd.to_datetime(ch.DECAY_DATE)
fechas = pd.date_range('2007-01-11', '2026-09-30', freq='30D')
obs = [((ch.DECAY_DATE.isna()) | (ch.DECAY_DATE > x)).sum() for x in fechas]
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(fechas, obs, color=INK, lw=2.2, label='Observado (catálogo)')
escala = ch.DECAY_DATE.isna().sum()
for nombre, etiqueta, col, ls in [('observado', 'Modelo calibrado, actividad solar predicha', '#2a78d6', '-'),
                                  ('70', 'Modelo calibrado, Sol tranquilo (F10.7 = 70)', '#1baf7a', '--'),
                                  ('200', 'Modelo calibrado, Sol activo (F10.7 = 200)', '#eb6834', '--'),
                                  ('observado_sin_calibrar', 'Sin calibrar (12.74·B*)', GR, ':')]:
    p = pd.read_csv(c.RES / f'proyeccion_fengyun1c_{nombre}.csv', parse_dates=['fecha'])
    ax.plot(p.fecha, p.en_orbita / p.en_orbita.iloc[0] * escala, color=col, lw=1.8, ls=ls, label=etiqueta)
ax.axvline(pd.Timestamp('2026-10-01'), color=GR, lw=1); ax.text(pd.Timestamp('2027-06-01'), 3350, 'hoy', fontsize=8.5, color=INK2)
ax.set_ylabel('Fragmentos de Fengyun-1C en órbita'); ax.set_ylim(0, 3700)
ax.set_title('Fragmentos chinos: pasado observado y proyección del modelo')
ax.legend(frameon=False, fontsize=8.3, loc='lower left'); estilo(ax)
guardar(fig, 'fig4_proyeccion_china.png')

# 5. Vida orbital contra altitud
cv = pd.read_csv(c.RES / 'vida_contra_altitud.csv').replace(np.inf, np.nan)
c16, c50, c84 = [[k for k in cv.columns if k.startswith(f'vida_anios_{q}')][0] for q in ('p16', 'p50', 'p84')]
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.fill_between(cv.altitud_km, cv[c84], cv[c16], color='#2a78d6', alpha=.15, lw=0, label='Rango del 68 % de los fragmentos')
ax.plot(cv.altitud_km, cv[c50], color='#2a78d6', lw=2, label='Fragmento típico (C$_D$A/m calibrado mediano)')
for r in sup.itertuples():
    if pd.notna(r.vida_mediana_anios):
        ax.scatter([r.altitud_km], [r.vida_mediana_anios], color=COL[r.evento], s=60, zorder=5, edgecolor='white')
        ax.text(r.altitud_km + 8, r.vida_mediana_anios * 1.3, f'{r.evento} (observado)', fontsize=8.3, color=COL[r.evento])
ax.annotate('', xy=(865, 150), xytext=(865, 19.7), arrowprops=dict(arrowstyle='->', color=COL['China 2007'], lw=1.5))
ax.scatter([865], [19.7], color=COL['China 2007'], s=60, zorder=5, edgecolor='white')
ax.text(855, 30, 'China 2007:\n65 % sigue\ntras 19.7 años', fontsize=8.3, color=COL['China 2007'], ha='right')
ax.axhline(5, color=GR, lw=1, ls=':'); ax.text(255, 5.6, '5 años', fontsize=8, color=INK2)
ax.set_yscale('log'); ax.set_ylim(2e-3, 300); ax.set_xlim(240, 900)
ax.set_yticks([0.01, 0.1, 1, 10, 100]); ax.set_yticklabels(['4 días', '1 mes', '1 año', '10 años', '100 años'])
ax.set_xlabel('Altura inicial (órbita circular, km)'); ax.set_ylabel('Vida orbital')
ax.set_title('Vida orbital de un fragmento contra altura (modelo) y vidas medianas observadas')
ax.legend(frameon=False, fontsize=8.3, loc='upper left'); estilo(ax)
guardar(fig, 'fig5_vida_altitud.png')

# 6. Densidad espacial y riesgo
de = pd.read_csv(c.RES / 'densidad_espacial.csv')
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
for col, colr, lw in [('Total', INK, 2.2), ('Prueba ASAT China 2007', '#2a78d6', 1.8), ('Colisión accidental 2009', '#eda100', 1.6)]:
    a1.plot(de.altitud_km, de[col] * 1e9, color=colr, lw=lw, label={'Total': 'Todos los objetos catalogados'}.get(col, col))
a1.set_yscale('log'); a1.set_ylim(1e-2, None); a1.set_xlabel('Altura (km)'); a1.set_ylabel('Densidad espacial (objetos por 10⁹ km³)')
a1.set_title('A. Densidad espacial por capa de 10 km'); a1.legend(frameon=False, fontsize=8.3); estilo(a1)
a2.plot(de.altitud_km, 100 * de['Prueba ASAT China 2007'] / de.Total, color='#2a78d6', lw=2, label='Prueba ASAT China 2007')
a2.plot(de.altitud_km, 100 * de['Colisión accidental 2009'] / de.Total, color='#eda100', lw=1.6, label='Colisión accidental 2009')
a2.set_xlabel('Altura (km)'); a2.set_ylabel('% del riesgo de colisión'); a2.set_xlim(300, 1500)
a2.set_title('B. Parte del riesgo de colisión debida a cada evento'); a2.legend(frameon=False, fontsize=8.3); estilo(a2)
guardar(fig, 'fig6_riesgo_densidad.png')

# 7. Riesgo que deja China a 800 km, en el tiempo
t = pd.read_csv(c.RES / 'riesgo_fengyun1c_tiempo.csv', parse_dates=['fecha'])
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(t.fecha, t.choques_anio_10m2_800km * 1e5, color='#2a78d6', lw=2)
ax.set_ylabel('Choques por año × 10⁵\n(satélite de 10 m² a 800 km)'); ax.set_ylim(0, None)
ax.set_title('Riesgo de colisión que dejan los fragmentos de Fengyun-1C a 800 km (modelo calibrado)'); estilo(ax)
guardar(fig, 'fig7_riesgo_china_tiempo.png')
print('Figuras en', c.FIG)
