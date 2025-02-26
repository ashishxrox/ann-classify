import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load encoders and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app title
st.title('Customer Churn Prediction')

st.subheader("1. What is Customer Churn?")
st.write("Customer churn in a bank refers to when customers stop using the bank’s services, close their accounts, or switch to a competitor.")

st.subheader("2. Why Does It Happen?")
st.write("Common reasons include poor customer service, high fees, better offers from competitors, inconvenience in banking services, or dissatisfaction with digital banking experience.")

st.subheader("3. How to Reduce Churn?")
st.write("Banks can reduce churn by improving customer service, offering personalized financial products, reducing fees, enhancing digital banking features, and maintaining strong customer relationships.")



# User input
geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 29)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cs_card = st.selectbox("Has Credit Card?", [0, 1])  # Fixed valid values
is_active_member = st.selectbox("Is Active Member?", [0, 1])

# Encode categorical inputs
gender_encoded = label_encoder_gender.transform([gender])[0]
geo_encoded = one_hot_encoder_geo.transform([[geography]])  # Remove `.toarray()`

# Convert input data to DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cs_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Convert geo_encoded to DataFrame
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

# Concatenate input data with one-hot encoded geography
input_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale input data
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)
pred_proba = prediction[0][0]


st.write("The prediction is ->", pred_proba)

# Display result
if pred_proba >= 0.05:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")
