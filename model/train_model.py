import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def get_data():
    data = load_iris()
    x = data.data
    y = data.target
    
    # One-hot encoding
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(max(y)+1))
    y = label_binarizer.transform(y)
    
    # Split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, 
        test_size=0.2, 
        random_state=42
    )
    return x_train, x_test, y_train, y_test

def build_model(capas, activation, last_activation, input_dim):
    model = Sequential()
    for i, neuronas_capa in enumerate(capas):
        if i == 0:
            model.add(Dense(neuronas_capa, activation=activation, input_dim=input_dim))
        elif i == len(capas)-1:
            model.add(Dense(neuronas_capa, activation=last_activation))
        else:
            model.add(Dense(neuronas_capa, activation=activation))
    return model

if __name__ == "__main__":
    # Configuración
    EPOCHS = 100
    CAPAS = [64, 3]
    ACTIVATION = "relu"
    LAST_ACTIVATION = "softmax"
    
    # Obtener datos
    x_train, x_test, y_train, y_test = get_data()
    INPUT_DIM = x_train.shape[1]
    
    # Construir y compilar modelo
    model = build_model(CAPAS, ACTIVATION, LAST_ACTIVATION, INPUT_DIM)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entrenar
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        verbose=0
    )
    
    # Guardar modelo
    model.save("model/iris_model.keras")
    print("Modelo entrenado y guardado correctamente.")