# Piloto: fragmentos de pruebas antisatélite, leyes de Kepler y riesgo de colisión

Proyecto piloto reproducible para el tema *"Persistencia orbital de los fragmentos de pruebas
antisatélite: una evaluación física de las normas internacionales sobre el espacio"*.
Autor: Luis Alberto Román Verde (Lic. Física / Lic. Relaciones Internacionales, UdeG).

**Pregunta:** ¿cuánto tiempo permanecen en órbita los fragmentos de una prueba antisatélite
(ASAT) según la altura a la que se hizo, y qué riesgo de colisión dejan para los satélites
de otros países?

## Casos

| Prueba | Satélite destruido | Altura | Designador |
|---|---|---|---|
| China, 11-ene-2007 | Fengyun-1C | ~865 km | 1999-025 |
| EE. UU., 21-feb-2008 | USA-193 | ~247 km | 2006-057 |
| India, 27-mar-2019 | Microsat-R | ~283 km | 2019-006 |
| Rusia, 15-nov-2021 | Kosmos 1408 | ~480 km | 1982-092 |

Las alturas son valores aproximados de referencia. Los datos de cada fragmento salen del catálogo.

## Cómo reproducirlo

```bash
pip install -r requirements.txt
python ejecutar_todo.py              # usa los datos guardados en datos_crudos/ (mismas cifras)
python ejecutar_todo.py --descargar  # descarga datos actuales (las cifras cambiarán un poco)
```

## Trazabilidad

**Tablas de datos crudos (Excel):** `datos_crudos_piloto_ASAT.xlsx` (fuentes con SHA-256, resumen con fórmulas,
fragmentos de las 4 pruebas y elementos orbitales) y `datos_crudos_clima_espacial.xlsx` (actividad solar diaria).
Las genera `07_tabla_datos_crudos.py` sin modificar los valores de los archivos originales.

- `datos_crudos/` guarda los archivos **tal como se descargaron**.
- `datos_crudos/manifiesto.json` registra la URL, la fecha y hora de descarga y la huella SHA-256 de cada uno.
- Cada cifra de este documento sale de un archivo en `resultados/`, generado por un script numerado.
- `resultados/huellas_resultados.json` guarda la huella SHA-256 de cada resultado.
- `test_piloto.py` comprueba cada fórmula física contra un resultado conocido a mano (7 pruebas).

| Archivo crudo | Fuente | Qué contiene |
|---|---|---|
| `satcat.csv` | CelesTrak | Catálogo de todos los objetos: lanzamiento, reingreso, perigeo, apogeo |
| `gp_fengyun1c_1999-025.csv` | CelesTrak | Elementos orbitales actuales de los fragmentos chinos (n, e, B*, dn/dt) |
| `gp_kosmos1408_1982-092.csv` | CelesTrak | Lo mismo, fragmentos rusos que quedan |
| `SW-All.csv` | CelesTrak | Actividad solar F10.7 e índice geomagnético Ap, observados y predichos |

---

## Metodología paso a paso

### Paso 1 — Descargar los datos (`01_descarga.py`)
Se descargan los cuatro archivos y se registra su huella digital. Nada se modifica a mano.

### Paso 2 — Leyes de Kepler (`02_kepler.py`)
Para cada fragmento, el catálogo da el **movimiento medio** *n* (vueltas por día) y la **excentricidad** *e*.

1. **Tercera ley** (tamaño de la órbita): n²a³ = μ  →  **a = (μ/n²)^(1/3)**, con μ = 398,600.44 km³/s².
2. **Primera ley** (órbita elíptica): perigeo r_p = a(1−e), apogeo r_a = a(1+e). La altura es r − R_T, con R_T = 6,378.137 km.
3. **Energía (vis-viva):** v² = μ(2/r − 1/a). El fragmento va más rápido en el perigeo que en el apogeo.
4. **Segunda ley** (áreas iguales en tiempos iguales) mediante la **ecuación de Kepler**, M = E − e·sen E: da la fracción del tiempo que el fragmento pasa en cada altura. Se usa en el paso 5.
5. **Control:** las alturas y periodos calculados se comparan con los que publica el catálogo.

**Resultado:** para los 1,982 fragmentos chinos con elementos publicados, las diferencias con el
catálogo son menores de 0.6 km en altura y de 0.005 min en periodo (95 % de los casos).
El cálculo es correcto. Con estos valores se construye el **diagrama de Gabbard** (figura 1):
la explosión de 2007 dejó perigeos entre ~550 y ~845 km y apogeos hasta más de 2,000 km.

### Paso 3 — Supervivencia observada (`03_supervivencia.py`)
Para cada prueba se cuentan los fragmentos catalogados **después** de la prueba (filtro por número de
catálogo) y cuántos siguen en órbita a lo largo del tiempo (figura 2).

| Prueba | Fragmentos | Vida mediana | En órbita hoy |
|---|---|---|---|
| EE. UU. 2008 (247 km) | 174 | 27 días | 0 (último reingreso: oct. 2009) |
| India 2019 (283 km) | 129 | 132 días | 0 (último reingreso: jun. 2022) |
| Rusia 2021 (480 km) | 1,805 | 192 días | 4 |
| China 2007 (865 km) | 3,533 | más de 19.7 años | **2,315 (65.5 %)** |

### Paso 4 — Modelo de decaimiento por frenado atmosférico (`04_decaimiento.py`)
El aire que queda a esas alturas frena al fragmento; su órbita se encoge hasta que cae.

- **Tamaño de la órbita:** con la energía orbital E = −μ/2a y la potencia del frenado −½ρBv³:
  **da/dt = −(a²/μ)·ρ·B·v³**, donde B = C_D·A/m (área entre masa).
- **Forma de la órbita:** **de/dt = −ρ·B·v·(e + cos ν)**. El frenado en el perigeo redondea la órbita.
- Ambas tasas se **promedian sobre una vuelta** usando la ecuación de Kepler (dt ∝ (1 − e cos E) dE).
- **Densidad del aire ρ:** modelo NRLMSISE-00 promediado sobre el globo, con la actividad solar real
  de cada fecha (el Sol activo calienta e "infla" la atmósfera).
- **Futuro:** actividad solar predicha hasta 2041; después se repite en ciclos de 11 años.
  Además hay dos escenarios de sensibilidad: Sol tranquilo (F10.7 = 70) y Sol activo (F10.7 = 200).

**Hallazgo del control (figura 3).** Si B se toma directamente del parámetro B* del catálogo
(B = 12.74·B*), el modelo predice una caída mucho más lenta que la observada. Es 38 veces más lenta a
800 km y solo 2 veces a 270 km. Esto ocurre porque el catálogo ajusta B* con su propia atmósfera
simplificada, que es demasiado densa a gran altura. Por eso **B se calibra para cada fragmento con su
tasa de caída observada** (dn/dt, publicada en el catálogo), usando dn/dt = −(3/2)(n/a)·da/dt (tercera ley).
El C_D·A/m calibrado tiene una mediana de **0.21 m²/kg**, un valor típico de fragmentos pequeños.

### Paso 4b — Verificación con datos que no se usaron para calibrar (`04b_verificacion.py`)

| Prueba | Observado | Modelo sin calibrar | Modelo calibrado |
|---|---|---|---|
| China: fracción que cae en una década | 20.9 % (2016–2026) | 1.3 % | **20.2 %** (2026–2036) |
| Rusia: vida mediana a 480 km | 192 días | — | 103 días (rango 52–379) |
| EE. UU. / India: vida mediana | 27 / 132 días | — | 2 / 3 días ✗ |

- El modelo calibrado reproduce bien China y Rusia.
- Falla en EE. UU. e India. La prueba usa órbitas circulares a la altura de la intercepción, pero
  en pruebas bajas la explosión lanza fragmentos a apogeos mucho más altos, donde duran más.
  Corregirlo requiere la órbita inicial de cada fragmento (historial de Space-Track; ver "Siguientes pasos").

**Predicción verificable:** los fragmentos rusos 50058 y 50621 deberían reingresar alrededor del
**9-nov-2026** y del **15-mar-2027**. Se puede comprobar en CelesTrak.

### Paso 5 — Análisis de riesgo de colisión (`05_riesgo.py`)
Método de **densidad espacial** o "gas cinético" (Kessler y Cour-Palais, 1978):

1. Con la ecuación de Kepler se calcula qué fracción del tiempo pasa **cada objeto en órbita** (29,319
   del catálogo) en cada capa de 10 km de altura.
2. **Densidad espacial:** S = Σ(fracción de tiempo) / volumen de la capa  [objetos/km³].
3. **Flujo** sobre un satélite de área A: **F = S · v_rel · A**, con v_rel = 10 km/s (valor típico).
4. **Probabilidad de choque** en un tiempo t: **P = 1 − exp(−F·t)**.

**Resultados** (satélite de 10 m², solo objetos catalogados, mayores de ~10 cm):

| Altura | Choques por año | Probabilidad en 5 años | Parte debida a la prueba china de 2007 |
|---|---|---|---|
| 420 km | 7.6 × 10⁻⁵ | 0.04 % | 0.4 % |
| 550 km | 1.2 × 10⁻⁴ | 0.06 % | 2.6 % |
| 800 km | 1.7 × 10⁻⁴ | 0.09 % | **24.9 %** |
| 850 km | 1.5 × 10⁻⁴ | 0.07 % | **36.2 %** |

Casi 20 años después, **una sola prueba explica una cuarta parte del riesgo de colisión catalogado a
800 km y más de un tercio a 850 km** (figura 6). Es más que la colisión accidental de 2009 entre
Iridium 33 y Kosmos 2251 (3.8 % a 800 km). Según el modelo calibrado, ese riesgo baja a la mitad
hacia 2050 y a una cuarta parte hacia 2100 (figura 7).

### Paso 6 — Figuras (`06_figuras.py`)
1. Diagrama de Gabbard (Kepler). 2. Supervivencia observada. 3. Calibración del frenado.
4. Pasado observado y proyección de los fragmentos chinos. 5. Vida orbital contra altura.
6. Densidad espacial y parte del riesgo por evento. 7. Riesgo chino a 800 km en el tiempo.

---

## Lectura para Relaciones Internacionales (resultado preliminar)

- **La altura decide quién paga.** Con el fragmento típico calibrado, la vida orbital es de días a
  300 km, de ~1 año hacia 550 km, de ~3 años a 600 km y de décadas por encima de 750 km (figura 5).
  Las pruebas de EE. UU. y la India limpiaron su órbita en meses o pocos años. La de China sigue
  imponiendo riesgo a todos los operadores casi 20 años después.
- **Una regla basada solo en la altura necesita margen.** Según el modelo, para que un fragmento típico
  caiga en menos de 5 años la intercepción tendría que ser por debajo de ~600 km. Además, la explosión
  lanza parte de los restos más arriba (figura 1), como muestra el caso de la India.
- Esto conecta con el artículo IX del Tratado del Espacio Ultraterrestre (1967), que pide tener en cuenta
  los intereses de los demás, y con la resolución 77/41 de la Asamblea General de la ONU (2022),
  que pide no hacer pruebas ASAT destructivas de ascenso directo.

## Limitaciones (reportarlas siempre)

- Solo se cuentan objetos **catalogados** (mayores de ~10 cm). Los fragmentos de 1–10 cm, más numerosos,
  no están, así que el riesgo real es **mayor**: las cifras son un límite inferior.
- El riesgo ignora la distribución en latitud y usa v_rel = 10 km/s fija.
- La calibración con dn/dt depende de la calidad de ese dato del catálogo. La verificación decenal de
  China la respalda, pero es una comparación entre décadas y poblaciones distintas.
- La atmósfera se promedia sobre el globo y sobre el año (equinoccio). No se incluye la presión de
  radiación solar, que importa para fragmentos muy ligeros a gran altura.
- Solo 1,932 de los 2,315 fragmentos chinos en órbita tienen elementos y dn/dt utilizables. La proyección
  de riesgo del paso 5 usa esos 1,932; para el total habría que multiplicar por ~1.2.

## Siguientes pasos para el proyecto completo

1. Registrarse en **Space-Track.org** (gratis) para obtener el **historial** de elementos de cada
   fragmento desde el día de cada prueba. Eso permite:
   - hacer los diagramas de Gabbard del momento de la explosión;
   - integrar cada fragmento desde su órbita inicial real;
   - validar la predicción fragmento por fragmento, que es lo que hoy falla en EE. UU. e India.
2. Parte de RI: Tratado de 1967, Convenio de Responsabilidad de 1972, directrices de la ONU sobre
   desechos, y las votaciones de la resolución 77/41 (incluida la posición de México).

## Referencias

- Kessler, D. J. y Cour-Palais, B. G. (1978). Collision frequency of artificial satellites: the creation
  of a debris belt. *Journal of Geophysical Research*, 83(A6), 2637–2646.
- Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications*, 4.ª ed. (SGP4, B*, frenado).
- Picone, J. M. et al. (2002). NRLMSISE-00 empirical model of the atmosphere. *Journal of Geophysical Research*, 107(A12).
- Kelso, T. S. CelesTrak (catálogo SATCAT, elementos GP, datos de clima espacial). https://celestrak.org
- ONU, Asamblea General, resolución A/RES/77/41 (2022); Tratado sobre el Espacio Ultraterrestre (1967).

Antes de citar, verifica volumen y páginas de cada referencia en la fuente original.
