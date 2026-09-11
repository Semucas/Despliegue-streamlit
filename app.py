# Interfaz gráfica

# Se crea interfaz gráfica con streamlit para captura de los datos

import streamlit as st
import pandas as pd
import pickle

st.title('Predicción de inversión en una tienda de videojuegos')

# Cargar el modelo entrenado (generado por el notebook videojuegos_prediccion.ipynb)
with open('modelo_videojuegos.pkl', 'rb') as f:
    modelo = pickle.load(f)

Edad = st.slider('Edad', min_value=14, max_value=52, value=20, step=1)
videojuego = st.selectbox('Videojuego', ["'Mass Effect'", "'Battlefield'", "'Fifa'", "'KOA: Reckoning'", "'Crysis'", "'Sim City'", "'Dead Space'", "'F1'"])
Plataforma = st.selectbox('Plataforma', ["'Play Station'", "'Xbox'", "PC", "Otros"])
Sexo = st.selectbox('Sexo', ['Hombre', 'Mujer'])
Consumidor_habitual = st.selectbox('Consumidor_habitual', ['True', 'False'])

# Dataframe
datos = [[Edad, videojuego, Plataforma, Sexo, Consumidor_habitual]]
data = pd.DataFrame(datos, columns=['Edad', 'videojuego', 'Plataforma', 'Sexo', 'Consumidor_habitual'])  # Dataframe con los mismos nombres de variables

# El modelo se entrenó con Consumidor_habitual como 0/1, así que se convierte
# el string del selectbox al mismo formato antes de predecir
data_modelo = data.copy()
data_modelo['Consumidor_habitual'] = (data_modelo['Consumidor_habitual'] == 'True').astype(int)

# Predicciones
st.header('Predicciones')

prediccion = modelo.predict(data_modelo)
data['Prediction'] = prediccion

st.dataframe(data)

# Recordar medida de error del modelo
st.warning("El modelo tiene un error del 12% (mape: error porcentual)")
