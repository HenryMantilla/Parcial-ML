# 🧠 ML Playground — Algoritmos con Streamlit

Este proyecto es una aplicación interactiva construida con **[Streamlit](https://streamlit.io/)** que permite **entrenar, visualizar y evaluar modelos de Machine Learning** de manera sencilla. Incluye tanto clasificadores estándar de `scikit-learn` como una implementación propia de **árboles de decisión con *Gain Ratio (C4.5)***.

---

## 🚀 Funcionalidades principales

- **Carga de datos**:  
  - Subir un archivo CSV propio.  
  - Usar el *DataFrame* compartido desde la página de Visualización.

- **Modelos disponibles**:
  - Árbol de Decisión (`gini`, `entropy`, `gain_ratio`).  
  - Random Forest.  
  - Gradient Boosting.  
  - Bagging (con árbol base personalizable).

- **Implementación propia**:  
  - Árbol de Decisión con *Gain Ratio (C4.5)* optimizado:  
    - Binning global por cuantiles o uniforme.  
    - Submuestreo de atributos.  
    - Manejo eficiente de variables categóricas con agrupamiento de categorías raras.  

- **Evaluación automática**:
  - Matriz de confusión.  
  - Métricas: Accuracy, Precision, Recall, F1, AUC.  
  - Reporte de clasificación en tabla interactiva.  
  - Curva ROC (para problemas binarios).  

- **Hiperparámetros ajustables desde la interfaz**:
  - Profundidad máxima.  
  - Número de estimadores.  
  - Estrategias de muestreo y división.  
  - Parámetros avanzados específicos de cada clasificador.  

---

## 🛠️ Requisitos

- Python 3.9+
- Dependencias principales:
  - `streamlit`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `seaborn`
  - `matplotlib`

Instalación rápida:

```bash
pip install -r requirements.txt
