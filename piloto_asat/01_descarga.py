"""Paso 1. Descarga de datos crudos con registro de trazabilidad.

Cada archivo se guarda sin modificar en datos_crudos/ y se registra en
datos_crudos/manifiesto.json con URL, fecha/hora UTC de descarga, tamaño y SHA-256.
"""
import datetime as dt, hashlib, json, time, urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
CRUDOS = RAIZ / 'datos_crudos'
FUENTES = {
    # Catálogo completo de objetos (fechas de lanzamiento/reingreso, perigeo, apogeo, periodo).
    'satcat.csv': 'https://celestrak.org/pub/satcat.csv',
    # Elementos orbitales actuales (formato OMM/CSV) de los fragmentos aún en órbita.
    'gp_fengyun1c_1999-025.csv': 'https://celestrak.org/NORAD/elements/gp.php?INTDES=1999-025&FORMAT=csv',
    'gp_kosmos1408_1982-092.csv': 'https://celestrak.org/NORAD/elements/gp.php?INTDES=1982-092&FORMAT=csv',
    # Índices de actividad solar y geomagnética observados (F10.7, Ap) que usa NRLMSISE-00.
    'SW-All.csv': 'https://celestrak.org/SpaceData/SW-All.csv',
}

def bajar(url, destino, intentos=5):
    for k in range(intentos):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                datos = r.read()
            destino.write_bytes(datos)
            return datos
        except Exception as ex:
            print(f'  intento {k + 1} falló: {ex}')
            time.sleep(2 ** (k + 1))
    raise RuntimeError(f'No se pudo descargar {url}')

def main():
    CRUDOS.mkdir(exist_ok=True)
    manifiesto = []
    for nombre, url in FUENTES.items():
        print('Descargando', nombre)
        datos = bajar(url, CRUDOS / nombre)
        manifiesto.append(dict(archivo=nombre, url=url,
                               descargado_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                               bytes=len(datos), sha256=hashlib.sha256(datos).hexdigest()))
    (CRUDOS / 'manifiesto.json').write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(manifiesto, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
