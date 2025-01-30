import pytest
import json
import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from api.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    response = client.get('/hc')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'OK'
    assert data['model_loaded'] is True

def test_predict_valid(client):
    sample_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post('/predict', json=sample_data)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['species'] == 'setosa'
    assert data['probabilities']['setosa'] > 0.9

def test_predict_invalid(client):
    # Campo faltante
    response = client.post('/predict', json={"sepal_length": 5.1})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "Campos faltantes" in data['error']

    # Valor no numérico
    response = client.post('/predict', json={
        "sepal_length": "cinco",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "debe ser un número (no cadena)" in data['error']