
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Загрузка модели и энкодера
model = joblib.load("model_xgb.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("Прогноз послеоперационных осложнений")

st.write("Введите послеоперационные параметры пациента:")

# Поля для ввода
features = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10', 'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15', 'feature_16', 'feature_17', 'feature_18', 'feature_19']
user_input = {}
for feat in features:
    user_input[feat] = st.number_input(feat, value=0.0)

X = pd.DataFrame([user_input])

# Предсказание
probs = model.predict_proba(X)[0]
pred_class = encoder.inverse_transform([np.argmax(probs)])[0]

st.subheader("Прогноз:")
st.write(f"Тип осложнения: **{pred_class}**")

st.subheader("Вероятности по классам:")
for cls, prob in zip(encoder.classes_, probs):
    st.write(f"Класс {cls}: {prob:.2%}")
