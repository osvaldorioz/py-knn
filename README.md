
El algoritmo **K-Nearest Neighbors (KNN)** es un método de clasificación supervisado basado en la similitud entre ejemplos. Funciona de la siguiente manera:  

1. **Cálculo de distancia**: Para cada punto de prueba, se calcula la distancia a todos los puntos de entrenamiento (comúnmente usando la distancia euclidiana).  
2. **Selección de vecinos**: Se seleccionan los `k` puntos más cercanos.  
3. **Votación mayoritaria**: Se elige la clase más frecuente entre los `k` vecinos para asignarla al punto de prueba.  

KNN es un algoritmo basado en memoria (lazy learning), ya que no construye un modelo explícito, sino que hace predicciones comparando directamente con los datos de entrenamiento.  

---

### Implementación en C++

En la implementación en C++ con Pybind11:  

1. **Entrada de datos**:  
   - Los datos de entrenamiento (`X_train` y `y_train`) y de prueba (`X_test`) se reciben como `py::array_t<>` desde Python.  

2. **Cálculo de distancias**:  
   - Se usa la norma Euclidiana para calcular la distancia entre cada punto de prueba y los puntos de entrenamiento.  

3. **Selección de vecinos más cercanos**:  
   - Se ordenan los índices de los datos de entrenamiento según la distancia a cada punto de prueba.  
   - Se toman los `k` índices más cercanos.  

4. **Votación para clasificación**:  
   - Se cuenta la frecuencia de cada etiqueta en los `k` vecinos.  
   - Se selecciona la etiqueta con mayor frecuencia.  

5. **Retorno de resultados**:  
   - La función devuelve un `py::array_t<int>` con las etiquetas predichas.  
