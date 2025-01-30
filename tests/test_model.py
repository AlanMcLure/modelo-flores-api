import numpy as np
import pytest
from tensorflow.keras.models import load_model
from pathlib import Path

def test_model_predictions():
    # Cargar el modelo
    model_path = Path("model/iris_model.keras")
    if not model_path.exists():
        pytest.fail("Modelo no encontrado en la ruta especificada")
    
    model = load_model(model_path)
    
    # Ejemplos de prueba (usando datos conocidos del dataset Iris)
    test_cases = [
        # setosa (debería predecir clase 0)
        {
            "input": [[5.1, 3.5, 1.4, 0.2]],
            "expected_class": 0,
            "expected_proba": [0.90, 0.09, 0.01]  # Ajustar según tu modelo
        },
        # versicolor (clase 1)
        {
            "input": [[6.0, 2.7, 4.5, 0.7]],
            "expected_class": 1,
            "expected_proba": [0.05, 0.8, 0.15]  # Valores más realistas
        },
        # virginica (clase 2)
        {
            "input": [[6.7, 3.0, 5.5, 2.5]],
            "expected_class": 2,
            "expected_proba": [0.05, 0.15, 0.8]
        }
    ]
    
    for case in test_cases:
        # Hacer predicción
        prediction = model.predict(np.array(case["input"]), verbose=0)[0]
        predicted_class = np.argmax(prediction)
        
        # Verificar clase
        assert predicted_class == case["expected_class"], \
            f"Error en {case['input']}: Esperado {case['expected_class']}, Obtenido {predicted_class}"
        
        # Verificar probabilidades (aproximadamente)
        for i, expected in enumerate(case["expected_proba"]):
            assert abs(prediction[i] - expected) < 0.1, \
                f"Probabilidad clase {i} fuera de rango: {prediction[i]:.2f}"