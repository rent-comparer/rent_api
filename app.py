import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import glob
import joblib
import re

app = Flask(__name__)
cors = CORS(app)
app.config["DEBUG"] = True

# --- Modelos cargados por provincia ---
path = 'api/models/'
files = glob.glob(os.path.join(path, "lr_*.pkl"))

MODELS = {}

# Diccionario de reemplazo fuera del bucle para eficiencia
REPLACEMENTS = {
    'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u',
    'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
    'ñ': 'n' # Opcional: si prefieres convertir ñ a n
}

def clean_text(text):
    text = text.lower()
    for accented, unaccented in REPLACEMENTS.items():
        text = re.sub(accented, unaccented, text)
    return text

for f in files:
    # 1. Obtener el nombre del archivo SIN modificar la ruta 'f'
    filename = os.path.basename(f)
    
    # 2. Limpiar el nombre para crear la clave (ej. 'lr_Ávila.pkl' -> 'avila')
    # Primero quitamos prefijos/sufijos
    province_raw = filename.replace("lr_", "").replace(".pkl", "")
    
    # Luego aplicamos la limpieza de tildes
    province_key = clean_text(province_raw)

    try:
        # 3. Cargar usando la ruta original 'f' (que sí existe en el disco)
        MODELS[province_key] = joblib.load(f)
        print(f"✅ Cargado: {province_key} (desde {filename})")
    except Exception as e:
        print(f"❌ Error cargando {filename}: {e}")
if not MODELS:
    raise Exception("No se cargaron modelos. Revisa la carpeta api/models/")
        
@app.route('/')
def home():
    return jsonify({'message': 'Bienvenidos a la API de Valora!'})

@app.route('/prediction', methods=['GET'])
def prediction():
    try:
        # --- Parámetros numéricos obligatorios ---
        surface   = int(request.args.get('surface'))
        provincia = request.args.get('provincia')        
        
        if not provincia:
             return jsonify({'error': 'Falta el parámetro provincia'}), 400
        
        # bedrooms  = int(request.args.get('bedrooms'))
        # restrooms = int(request.args.get('restrooms'))

        # # --- Parámetros booleanos (0 o 1), por defecto 0 ---
        # terraza    = int(request.args.get('terraza',    0))
        # ascensor   = int(request.args.get('ascensor',   0))
        # piscina    = int(request.args.get('piscina',    0))
        # calefacion = int(request.args.get('calefaccion', 0))

    except (TypeError, ValueError):
        return jsonify({'error': 'Parámetro inválido o faltante'}), 400
    


    # --- Selección de modelo por provincia ---

    provincia_key = provincia.lower() 
        
    if provincia_key not in MODELS:
        return jsonify({
            'error': f"Provincia '{provincia}' no encontrada. Opciones: {list(MODELS.keys())}"
        }), 400

    model = MODELS[provincia]

    # --- Predicción ---
    input_data = [[surface]] #, bedrooms, restrooms, terraza, ascensor, piscina, calefacion
    result = model.predict(input_data)

    return jsonify({
        'provincia':  provincia,
        'prediction': float(result[0]),
        'input': {
            'surface':     surface,
            # 'bedrooms':    bedrooms,
            # 'restrooms':   restrooms,
            # 'Terraza':     terraza,
            # 'Ascensor':    ascensor,
            # 'Piscina':     piscina,
            # 'Calefacción': calefacion,
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")