from flask import Flask, jsonify, request
from flask_cors import CORS
import os, glob, joblib, re, pickle
import pandas as pd

app = Flask(__name__)
cors = CORS(app)
app.config["DEBUG"] = True

# ── Modelos de Alejandro — un pkl por provincia (para /prediction) ──────────
path  = 'api/models/'
files = glob.glob(os.path.join(path, "lr_*.pkl"))

MODELS = {}
REPLACEMENTS = {
    'á':'a','é':'e','í':'i','ó':'o','ú':'u','ü':'u',
    'à':'a','è':'e','ì':'i','ò':'o','ù':'u','ñ':'n'
}

def clean_text(text):
    text = text.lower()
    for accented, unaccented in REPLACEMENTS.items():
        text = re.sub(accented, unaccented, text)
    return text

for f in files:
    filename    = os.path.basename(f)
    province_key = clean_text(filename.replace("lr_", "").replace(".pkl", ""))
    try:
        MODELS[province_key] = joblib.load(f)
        print(f"✅ Cargado: {province_key}")
    except Exception as e:
        print(f"❌ Error cargando {filename}: {e}")

if not MODELS:
    raise Exception("No se cargaron modelos. Revisa la carpeta api/models/")

# ── Modelo enriquecido — para /compare (superficie + features + provincia) ──
with open('api/models/model_enriquecido.pkl', 'rb') as f:
    MODEL_COMPARE = pickle.load(f)

df_ine  = pd.read_csv('api/data/ine_provincias_2022.csv')
INE_MAP = df_ine.set_index('provincia')[
    ['renta_neta_persona_2022', 'tasa_paro_q4_2022']
].to_dict('index')
PROVINCIAS = sorted(df_ine['provincia'].tolist())

print(f"✅ model_enriquecido.pkl cargado. {len(PROVINCIAS)} provincias disponibles.")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return jsonify({'message': 'Bienvenidos a la API de Valora!'})


@app.route('/prediction', methods=['GET'])
def prediction():
    """Tasador de Alejandro — predice precio por superficie y provincia."""
    try:
        surface  = int(request.args.get('surface'))
        provincia = request.args.get('provincia')
        if not provincia:
            return jsonify({'error': 'Falta el parámetro provincia'}), 400
    except (TypeError, ValueError):
        return jsonify({'error': 'Parámetro inválido o faltante'}), 400

    provincia_key = clean_text(provincia)
    if provincia_key not in MODELS:
        return jsonify({'error': f"Provincia '{provincia}' no encontrada."}), 400

    result = MODELS[provincia_key].predict([[surface]])
    return jsonify({
        'provincia':  provincia,
        'prediction': float(result[0]),
        'input':      {'surface': surface}
    })


@app.route('/compare', methods=['GET'])
def compare():
    """
    Comparador interprovincial — dado un piso con todas sus características,
    predice cuánto costaría en cada provincia de España.

    Parámetros (GET):
        surface          (int,   obligatorio) — superficie en m²
        provincia        (str,   obligatorio) — provincia de origen
        bedrooms         (int,   opcional, default 2)
        restrooms        (int,   opcional, default 1)
        terraza          (0/1,   opcional, default 0)
        ascensor         (0/1,   opcional, default 0)
        piscina          (0/1,   opcional, default 0)
        calefaccion      (0/1,   opcional, default 0)
        precio_publicado (float, opcional)  — para calcular sobrevaloración

    Ejemplo:
        GET /compare?surface=120&provincia=Bizkaia&bedrooms=3&restrooms=2
                    &terraza=1&ascensor=1&precio_publicado=1400
    """
    try:
        surface          = int(request.args.get('surface'))
        provincia        = request.args.get('provincia')
        bedrooms         = int(request.args.get('bedrooms',    2))
        restrooms        = int(request.args.get('restrooms',   1))
        terraza          = int(request.args.get('terraza',     0))
        ascensor         = int(request.args.get('ascensor',    0))
        piscina          = int(request.args.get('piscina',     0))
        calefaccion      = int(request.args.get('calefaccion', 0))
        precio_publicado = request.args.get('precio_publicado')

        if not provincia:
            return jsonify({'error': 'Falta el parámetro provincia'}), 400
        if precio_publicado is not None:
            precio_publicado = float(precio_publicado)

    except (TypeError, ValueError):
        return jsonify({'error': 'Parámetro inválido o faltante'}), 400

    # Verificar que la provincia de origen existe en nuestros datos INE
    if provincia not in INE_MAP:
        return jsonify({'error': f"Provincia '{provincia}' no encontrada."}), 400

    # Construir input para las 50 provincias — solo cambia location_name e INE
    rows = []
    for prov in PROVINCIAS:
        ine = INE_MAP[prov]
        rows.append({
            'surface':                 surface,
            'bedrooms':                bedrooms,
            'restrooms':               restrooms,
            'location_name':           prov,
            'Terraza':                 terraza,
            'Ascensor':                ascensor,
            'Piscina':                 piscina,
            'Calefacción':             calefaccion,
            'renta_neta_persona_2022': ine['renta_neta_persona_2022'],
            'tasa_paro_q4_2022':       ine['tasa_paro_q4_2022']
        })

    df_input = pd.DataFrame(rows)
    df_input['precio_predicho'] = MODEL_COMPARE.predict(df_input).round(2)

    ranking = df_input[['location_name', 'precio_predicho']]\
        .sort_values('precio_predicho', ascending=False)\
        .reset_index(drop=True)

    ranking_list = [
        {'provincia': str(r['location_name']), 'precio_predicho': float(r['precio_predicho'])}
        for _, r in ranking.iterrows()
    ]

    ranking_posicion = next(
        (i + 1 for i, r in enumerate(ranking_list) if r['provincia'] == provincia),
        None
    )

    response = {
        'input': {
            'surface': surface, 'provincia': provincia,
            'bedrooms': bedrooms, 'restrooms': restrooms,
            'terraza': terraza, 'ascensor': ascensor,
            'piscina': piscina, 'calefaccion': calefaccion
        },
        'ranking':           ranking_list,
        'ranking_posicion':  ranking_posicion,
        'ranking_total':     len(ranking_list)
    }

    # Diagnóstico de sobrevaloración (solo si se pasa precio_publicado)
    if precio_publicado is not None:
        precio_predicho_origen = next(
            (r['precio_predicho'] for r in ranking_list if r['provincia'] == provincia),
            None
        )
        if precio_predicho_origen:
            diferencia = precio_publicado - precio_predicho_origen
            porcentaje = (diferencia / precio_predicho_origen) * 100
            if porcentaje > 20:
                estado = 'SOBREVALORADO'
            elif porcentaje < -20:
                estado = 'INFRAVALORADO'
            else:
                estado = 'EN PRECIO DE MERCADO'

            response['diagnostico'] = {
                'precio_publicado': precio_publicado,
                'precio_predicho':  precio_predicho_origen,
                'diferencia_euros': round(diferencia, 2),
                'diferencia_pct':   round(float(porcentaje), 1),
                'estado':           estado
            }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")