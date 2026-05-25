import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)
app.config["DEBUG"] = True

# --- Modelos cargados por provincia ---
MODELS = {
    "low":    pd.read_pickle('api/models/modelo_xgb_g1.pkl'),
    "medium": pd.read_pickle('api/models/modelo_xgb_g2.pkl'),
    "high":  pd.read_pickle('api/models/modelo_xgb_g3.pkl'),
}
DEFAULT_MODEL = "medium"  # fallback si no se especifica provincia

@app.route('/')
def home():
    return jsonify({'message': 'Bienvenidos a la API de Valora!'})

# http://localhost:5000/prediction
#   ?surface=100&bedrooms=2&restrooms=1
#   &Terraza=1&Ascensor=1&Piscina=0&Calefacción=1
#   &provincia=barcelona

@app.route('/prediction', methods=['GET'])
def prediction():
    try:
        # --- Parámetros numéricos obligatorios ---
        surface   = int(request.args.get('surface'))
        bedrooms  = int(request.args.get('bedrooms'))
        restrooms = int(request.args.get('restrooms'))

        # --- Parámetros booleanos (0 o 1), por defecto 0 ---
        terraza    = int(request.args.get('terraza',    0))
        ascensor   = int(request.args.get('ascensor',   0))
        piscina    = int(request.args.get('piscina',    0))
        calefacion = int(request.args.get('calefaccion', 0))

    except (TypeError, ValueError):
        return jsonify({'error': 'Parámetro inválido o faltante'}), 400

    # --- Selección de modelo por provincia ---
    provincia = request.args.get('provincia', DEFAULT_MODEL)
    
    modelo_provincia = {
        # --- LOW ---
        "Albacete": "low",
        "Murcia": "low",
        "Cádiz": "low",
        "Soria": "low",
        "Valladolid": "low",
        "La_Rioja": "low",
        "Castelló": "low",
        "Huelva": "low",
        "Burgos": "low",
        "Córdoba": "low",
        "ACoruna": "low",
        "Asturias": "low",
        "Almería": "low",
        "Toledo": "low",
        "Cuenca": "low",
        "Huesca": "low",
        "Badajoz": "low",
        "Ávila": "low",
        "Cáceres": "low",
        "León": "low",
        "Ourense": "low",
        "Palencia": "low",
        "Lugo": "low",
        "Teruel": "low",
        "Jaén": "low",
        "Zamora": "low",
        "Ciudad_Real": "low",

        # --- MEDIUM ---
        "Araba": "medium",
        "Málaga": "medium",
        "Navarra": "medium",
        "Santa_Cruz_Tenerife": "medium",
        "Las_Palmas": "medium",
        "Sevilla": "medium",
        "Alacant": "medium",
        "Cantabria": "medium",
        "Tarragona": "medium",
        "Segovia": "medium",
        "Granada": "medium",
        "Guadalajara": "medium",
        "Zaragoza": "medium",
        "Salamanca": "medium",
        "Pontevedra": "medium",
        "Lleida": "medium",

        # --- HIGH ---
        "Illes_Balears": "high",
        "Barcelona": "high",
        "Madrid": "high",
        "Gipuzkoa": "high",
        "Bizkaia": "high",
        "Valencia": "high",
        "Girona": "high"
    }
    
    if provincia not in modelo_provincia:
        return jsonify({
            'error': f"Provincia '{provincia}' no soportada. Opciones: {list(MODELS.keys())}"
        }), 400

    #model = MODELS[provincia]
    model = MODELS[modelo_provincia[provincia]]

    # --- Predicción ---
    input_data = [[surface, bedrooms, restrooms, terraza, ascensor, piscina, calefacion]]
    result = model.predict(input_data)

    return jsonify({
        'provincia':  provincia,
        'prediction': float(result[0]),
        'input': {
            'surface':     surface,
            'bedrooms':    bedrooms,
            'restrooms':   restrooms,
            'Terraza':     terraza,
            'Ascensor':    ascensor,
            'Piscina':     piscina,
            'Calefacción': calefacion,
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")