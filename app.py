import streamlit as st
import numpy as np
import pickle

st.title("Employee Salary Prediction Model")
st.sidebar.header("Project Description")
st.sidebar.write("This app predicts whether an employee earns >50K or <=50K per year.")

st.sidebar.write("The objective of this project is to build a machine learning model that predicts whether an individual earns more than $50K per year based on various demographic and employment attributes. The prediction is based on the UCI Adult Income dataset, which includes features like age, education, occupation, marital status, and more.")
st.sidebar.write("This project aims to assist HR professionals and policy-makers in understanding income distribution and targeting eligible candidates based on demographic trends. The challenge lies in preprocessing the categorical data effectively and designing an interface for real-time predictions using a trained model.")
st.sidebar.write("The model categorizes individuals into two classes: <=50K and >50K income groups, making it useful for decision-making and classification tasks in business or government applications." )

# Load trained model and encoders
@st.cache_resource
def load_resources():
    clf = pickle.load(open("rf_model.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
    return clf, encoders

clf, encoders = load_resources()

# List your features as in training
input_features = [
    "age", "workclass", "fnlwgt", "educational-num", "marital-status",
    "occupation", "relationship", "race", "gender",
    "capital-gain", "capital-loss", "hours-per-week", "native-country"
]

# Taking input from user
def get_user_input(encoders):
    age = st.number_input("Age", 16, 100, 30)
    workclass = st.selectbox("Workclass", encoders["workclass"].classes_)
    fnlwgt = st.number_input("Fnlwgt", 10000, 1000000, 200000)
    educational_num = st.slider("Educational-num", 1, 16, 9)
    marital_status = st.selectbox("Marital Status", encoders["marital-status"].classes_)
    occupation = st.selectbox("Occupation", encoders["occupation"].classes_)
    relationship = st.selectbox("Relationship", encoders["relationship"].classes_)
    race = st.selectbox("Race", encoders["race"].classes_)
    gender = st.selectbox("Gender", encoders["gender"].classes_)
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
    hours_per_week = st.slider("Hours per week", 1, 99, 40)
    native_country = st.selectbox("Native Country", encoders["native-country"].classes_)
    return [
        age,
        encoders["workclass"].transform([workclass])[0],
        fnlwgt,
        educational_num,
        encoders["marital-status"].transform([marital_status])[0],
        encoders["occupation"].transform([occupation])[0],
        encoders["relationship"].transform([relationship])[0],
        encoders["race"].transform([race])[0],
        encoders["gender"].transform([gender])[0],
        capital_gain,
        capital_loss,
        hours_per_week,
        encoders["native-country"].transform([native_country])[0],
    ]

user_input = get_user_input(encoders)
input_np = np.array(user_input).reshape(1, -1)

if st.button("Predict Salary"):
    result = clf.predict(input_np)[0]
    pred_label = "<=50K" if result == 0 else ">50K"
    st.success(f"Predicted Salary Class: **{pred_label}**")
