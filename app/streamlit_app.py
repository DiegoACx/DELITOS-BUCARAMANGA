import streamlit as st
import numpy as np
import joblib

# Cargar el modelo y el escalador
scaler = joblib.load('standard_scalerjb.bin')
model = joblib.load('regresion.bin')

# Configurar el título de la aplicación
st.title("Predicción de Delitos en Bucaramanga")

# Crear selectores en la barra lateral para cada feature
st.sidebar.header("Selecciona los valores de entrada")

comuna = st.sidebar.selectbox(
    "Comuna", 
    options=list(range(1, 18)), 
    format_func=lambda x: ["NORTE", "NORORIENTAL", "SAN FRANCISCO", "OCCIDENTAL", "GARCIA ROVIRA", "LA CONCORDIA", "LA CIUDADELA", 
                          "SUROCCIDENTE", "LA PEDREGOSA", "PROVENZA", "SUR", "CABECERA DEL LLANO", "ORIENTAL", "MORRORICO", "CENTRO", 
                          "LAGOS DEL CACIQUE", "MUTIS"][x - 1]
)

mes = st.sidebar.selectbox("Mes", list(range(1, 13)))

sexo_victima = st.sidebar.selectbox(
    "Sexo de la Víctima", 
    options=[1, 2], 
    format_func=lambda x: "MASCULINO" if x == 1 else "FEMENINO"
)

movil_victima = st.sidebar.selectbox(
    "Móvil de la Víctima", 
    options=list(range(1, 8)), 
    format_func=lambda x: ["A PIE", "MOTOCICLETA", "VEHICULO", "BUS", "TAXI", "BICICLETA", "METRO"][x - 1]
)

dia_semana = st.sidebar.selectbox(
    "Día de la Semana", 
    options=list(range(1, 8)), 
    format_func=lambda x: ["DOMINGO", "LUNES", "MARTES", "MIERCOLES", "JUEVES", "VIERNES", "SÁBADO"][x - 1]
)

rango_horario = st.sidebar.selectbox(
    "Rango Horario", 
    options=list(range(1, 7)), 
    format_func=lambda x: ["MADRUGADA", "MAÑANA", "MEDIODIA", "TARDE", "ANOCHECER", "NOCHE"][x - 1]
)

cursovida_victima = st.sidebar.selectbox(
    "Curso de Vida de la Víctima", 
    options=list(range(1, 7)), 
    format_func=lambda x: ["PRIMERA INFANCIA", "INFANCIA", "ADOLESCENCIA", "JUVENTUD", "ADULTEZ", "PERSONA MAYOR"][x - 1]
)

# Crear un array con los valores seleccionados
input_data = np.array([[comuna, mes, sexo_victima, movil_victima, dia_semana, rango_horario, cursovida_victima]])

# Escalar los datos de entrada
scaled_data = scaler.transform(input_data)

# Realizar la predicción y calcular la probabilidad
prediction = model.predict(scaled_data)
prediction_probability = model.predict_proba(scaled_data)

# Mostrar los resultados
st.write("### Predicción del delito:")
st.write(f"Clase predicha: {prediction[0]}")
st.write("### Probabilidad de la predicción:")
st.write(f"Probabilidad por clase: {prediction_probability[0]}")
