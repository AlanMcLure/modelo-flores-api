import pathlib
import logging
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)

launch_time = time.monotonic_ns()

# Constantes
VALID_RANGES = {
    'sepal_length': (4.0, 8.0),
    'sepal_width': (2.0, 4.5),
    'petal_length': (1.0, 7.0),
    'petal_width': (0.1, 2.5)
}
SPECIES = ['setosa', 'versicolor', 'virginica']

try:
    # Ruta al modelo (desde app.py)
    model_path = pathlib.Path(__file__).parent.parent / "model" / "iris_model.keras"
    if not model_path.exists():
        raise FileNotFoundError("Modelo no encontrado en la ruta especificada")
        
    model = tf.keras.models.load_model(model_path)
    logging.info("Modelo cargado exitosamente")
    
except Exception as e:
    logging.critical(f"Error crítico al cargar el modelo: {str(e)}")
    exit(1)


@app.route('/')
def index():
    return render_template('index.html')

    
@app.route('/hc', methods=['GET'])
def health_check():
    try:
        current_time = time.monotonic_ns()
        uptime_seconds = (current_time - launch_time) / 1e9
        
        return jsonify({
            "status": "OK",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "uptime_sec": round(uptime_seconds, 2),
            "model_loaded": True
        })
    except Exception as e:
        logging.error(f"Error en health check: {str(e)}")
        return jsonify({"status": "ERROR"}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validar Content-Type
        if request.content_type != 'application/json':
            return jsonify({"error": "Se requiere Content-Type: application/json"}), 415
            
        data = request.get_json()
        
        # Obtener y validar estructura del JSON
        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify({"error": "El cuerpo debe ser un objeto JSON con las características de la flor"}), 400
        
        # Validar campos
        required = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        if any(field not in data for field in required):
            missing = [f for f in required if f not in data]
            return jsonify({"error": f"Campos faltantes: {', '.join(missing)}"}), 400
        
        # Validar y procesar valores
        features = []
        for field in required:
            try:
                # --- Versión estricta (solo acepta números) ---
                # No sé si debería no aceptar cadenas que puedan convertirse a números
                # Verificar que el valor es de tipo numérico (int/float)
                if not isinstance(data[field], (int, float)):
                    return jsonify({"error": f"{field} debe ser un número (no cadena)"}), 400
                
                value = float(data[field])
                features.append(value)
                
            except (TypeError, ValueError):
                return jsonify({"error": f"{field} debe ser numérico"}), 400
        
        # Validar rangos
        for i, (field, value) in enumerate(zip(required, features)):
            min_val, max_val = VALID_RANGES[field]
            if not (min_val <= value <= max_val):
                return jsonify({
                    "error": f"{field} debe estar entre {min_val} y {max_val} cm"
                }), 400
        
        # Predicción
        try:
            prediction = model.predict(np.array([features]), verbose=0)[0]
        except Exception as e:
            logging.error(f"Error en predicción: {str(e)}", exc_info=True)
            return jsonify({"error": "Error interno en modelo"}), 500
            
        # Formatear respuesta
        return jsonify({
            "species": SPECIES[np.argmax(prediction)],
            "probabilities": {
                species: round(float(prediction[i]), 4) 
                for i, species in enumerate(SPECIES)
            }
        })
        
    except Exception as e:
        logging.error(f"Error general: {str(e)}")
        return jsonify({"error": "Error interno del servidor"}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
