import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.title('Primary Aluminium Production and GHG Emissions Forecasting App')
st.write('This app provides Aluminium production and emissions forecasts based on various models.')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(data.head())

    y = data['All']  # Replace 'Target_Column' with your target column name
    X = data.drop(['All', 'Alumina Grade', 'Emission Source'], axis=1)  # Assuming all other columns are features

    model_choice = st.selectbox(
        'Select a forecasting model', 
        ['Linear Regression', 'XGBoost', 'Random Forest']
    )

    forecast_points = st.slider("Select number of future points for forecasting", 1, 100, 30)

    if st.button('Run Forecast'):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_choice == 'Linear Regression':
            model = LinearRegression()
            model.fit(X_train, y_train)
            forecast = model.predict(X_test[:forecast_points])
            
        elif model_choice == 'XGBoost':
            model = XGBRegressor()
            model.fit(X_train, y_train)
            forecast = model.predict(X_test[:forecast_points])
            
        elif model_choice == 'Random Forest':
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            forecast = model.predict(X_test[:forecast_points])
            
        mse = mean_squared_error(y_test[:forecast_points], forecast)
        st.write(f"Mean Squared Error: {mse}")
        
        st.write("Forecasted Values:")
        st.write(forecast)

        plt.figure(figsize=(12, 6))
        plt.plot(y_test[:forecast_points].values, label='Actual')
        plt.plot(forecast, label='Forecasted')
        plt.legend()
        st.pyplot(plt)

else:
    st.write('Please upload a data file to proceed.')