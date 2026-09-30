"""Ejecuta el piloto completo en orden. Uso:
    python ejecutar_todo.py              # usa los datos ya descargados (reproduce las cifras exactas)
    python ejecutar_todo.py --descargar  # vuelve a descargar datos actuales (las cifras cambiarán)
Al final registra el SHA-256 de cada resultado en resultados/huellas_resultados.json.
"""
import hashlib, json, subprocess, sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
pasos = (['01_descarga.py'] if '--descargar' in sys.argv else []) + [
    'test_piloto.py', '02_kepler.py', '03_supervivencia.py', '04_decaimiento.py',
    '04b_verificacion.py', '05_riesgo.py', '06_figuras.py']
for p in pasos:
    print(f'== {p}', flush=True)
    subprocess.run([sys.executable, str(RAIZ / p)], check=True, cwd=RAIZ)
huellas = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
           for f in sorted((RAIZ / 'resultados').iterdir()) if f.suffix in ('.csv', '.json')}
(RAIZ / 'resultados' / 'huellas_resultados.json').write_text(json.dumps(huellas, indent=2), encoding='utf-8')
print('Listo.')
