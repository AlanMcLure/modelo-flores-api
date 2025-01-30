# Predicción Flores API

Esta API permite clasificar especies de flores Iris (setosa, versicolor, virginica) basándose en medidas de sépalo y pétalo.

## Setup

### 1. Requisitos:
   - Python 3.8+
   - Entorno virtual recomendado

### 2. Instalación:
```bash
# Crear y activar entorno virtual (Windows)
python -m venv venv
.\venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Arrancar el proyecto:

```bash
# Configurar entorno para TensorFlow (Windows PowerShell)
$env:TF_ENABLE_ONEDNN_OPTS=0

# Arrancar el proyecto
cd .\api\
flask --app app run --debug
```

## Modelo

- Análisis del modelo disponible en jupyter notebook y en html
- Para entrenar el modelo:

```bash
# En la raíz del proyecto
python .\model\train_model.py
```

- Guardado como iris_model.keras

## Endpoints

### 1. Health Check (`/hc`)

| Método | Parámetros | Descripción                |
| :----- | :-------   | :------------------------- |
| `GET`  | `Ninguno`  | Estado del servidor (fecha y uptime)  |

- **Ejemplo de Response**:
  
  ```json
  {
    "model_loaded": true,
    "status": "OK",
    "timestamp": "2025-01-30T22:11:24.635385Z",
    "uptime_sec": 153.41
  }
  ```

### 2. Predict (`/predict`)

| Método | Parámetros | Descripción                |
| :----- | :-------   | :------------------------- |
| `POST`  | `JSON Body`  | Clasifica una flor Iris en una de las 3 especies (setosa, versicolor, virginica)  |

- **Ejemplo de Request**:
  ```json
  {
    "sepal_length": 5.1,
    "sepal_width": 3.4,
    "petal_length": 1.5,
    "petal_width": 0.1
  }
  ```

- **Ejemplo de Response**:
  ```json
  {
    "probabilities": {
        "setosa": 0.967,
        "versicolor": 0.0325,
        "virginica": 0.0004
    },
    "species": "setosa"
  }
  ```

### 3. Web (`/`)

| Método | Parámetros | Descripción                |
| :----- | :-------   | :------------------------- |
| `GET`  | `Ninguno`  | Página de Inicio del Cliente Web  |

- **Respuesta**: HTML renderizado de la página web
- **Ejemplo de Uso**:

  ```bash
  # Acceder desde navegador o curl
  curl http://localhost:5000/
  ```

## Testing

Hay tres opciones para hacer las pruebas de la API:

- Vía Postman importando el archivo Flores_API_Test.postman_collection.json
- Vía web navegando a http://localhost:5000/
- Vía Pytest (también hay test del modelo):
```bash
# En la raíz del proyecto
pytest .\tests\ -v
```
