import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('play_tennis_rf_model.joblib')
label_encoders = joblib.load('play_tennis_label_encoders.joblib')

# Streamlit app
st.title("🏸 Play Tennis Prediction App")

st.write("Fill in the conditions to predict if you should play tennis!")

# User inputs
outlook = st.selectbox("Outlook", label_encoders['Outlook'].classes_)
temperature = st.selectbox("Temperature", label_encoders['Temperature'].classes_)
humidity = st.selectbox("Humidity", label_encoders['Humidity'].classes_)
wind = st.selectbox("Wind", label_encoders['Wind'].classes_)

if st.button("Predict"):
    # Transform inputs using label encoders
    input_data = pd.DataFrame({
        'Outlook': [label_encoders['Outlook'].transform([outlook])[0]],
        'Temperature': [label_encoders['Temperature'].transform([temperature])[0]],
        'Humidity': [label_encoders['Humidity'].transform([humidity])[0]],
        'Wind': [label_encoders['Wind'].transform([wind])[0]]
    })

    # Make prediction
    prediction = model.predict(input_data)
    predicted_label = label_encoders['PlayTennis'].inverse_transform(prediction)[0]

    st.success(f"Prediction: **{predicted_label}**")

    # Show individual tree votes (optional)
    st.subheader("🌲 Individual Tree Predictions")
    for i, tree in enumerate(model.estimators_):
        tree_pred = tree.predict(input_data)
        tree_pred_int = tree_pred.astype(int)  # Cast to integer
        tree_label = label_encoders['PlayTennis'].inverse_transform(tree_pred_int)[0]
        st.write(f"Tree {i + 1}: {tree_label}")