from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import knn_module 
import json

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    
@app.post("/k-nearest-neighbors")
def calculo(samples: int, test_samples: int, n_neighbors: int):
    output_file = 'k_nearest_neighbors.png'
    
    # Generación de datos sintéticos
    np.random.seed(0)
    X_train = np.random.rand(samples, 2) * 10  # n puntos en 2D
    y_train = np.random.choice([0, 1], size=100)  # Etiquetas binarias

    X_test = np.random.rand(test_samples, 2) * 10  # n puntos de prueba en 2D
    k = n_neighbors  # Número de vecinos

    # Clasificación con KNN
    y_pred = knn_module.knn_classify(X_train, y_train, X_test, k)

    # Visualización de los datos
    plt.figure(figsize=(12, 5))

    # Gráfica de dispersión
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', label='Train Data', alpha=0.6)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k', marker='s', label='Test Data')
    plt.legend()
    plt.title("KNN - Clasificación de Datos")

    # Histograma de predicciones
    plt.subplot(1, 2, 2)
    plt.hist(y_pred, bins=np.arange(-0.5, 2, 1), rwidth=0.8, color='blue', alpha=0.7, edgecolor='black')
    plt.xticks([0, 1])
    plt.xlabel("Clase Predicha")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de Predicciones")

    plt.tight_layout()
    #plt.show()

    plt.savefig(output_file)
    plt.close()
    
    j1 = {
        "Grafica": output_file
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/k-nearest-neighbors-graph")
def getGraph(output_file: str):
    return FileResponse(output_file, media_type="image/png", filename=output_file)
