from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("new_dataset.csv", parse_dates=["Date"])

# Extract year and month as features
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Define the target variables
target_rainfall = 'Rainfall_mm'
target_temperature = 'Temperature_Celsius'
target_weather = 'Weather'
target_energy_demand = 'Energy Demand (GWh)'

# Features for prediction
features = ['Year', 'Month', 'Solar Energy (GWh)', 'Hydroelectric Power (GWh)', 'Energy Demand (GWh)', 'Total_maxpower (gwh)']

# Split the dataset into training data
train_data = data[data['Year'] < 2023]

# Predict Weather (Rainy/Sunny)
X = train_data[features]
y = train_data[target_weather]
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y)

# Predict Energy Demand
X = train_data[features]
y = train_data[target_energy_demand]
xgb_regressor = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
xgb_regressor.fit(X, y)

# Predict Temperature
X = train_data[features]
y = train_data[target_temperature]
temperature_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
temperature_regressor.fit(X, y)

# Predict Rainfall
X = train_data[features]
y = train_data[target_rainfall]
rainfall_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rainfall_regressor.fit(X, y)

# Function to predict for user input date
def predict_for_user_input(input_date):
    # Extract year and month from the input date
    input_year = input_date.year
    input_month = input_date.month
    
    # Use the previous year's data to predict
    previous_year_data = train_data[(train_data['Year'] == (input_year - 1)) & (train_data['Month'] == input_month)]
    input_features = previous_year_data[features].values.reshape(1, -1)
    
    # Predict Weather
    predicted_weather = rf_classifier.predict(input_features)[0]
    
    # Predict Energy Demand
    predicted_energy_demand = xgb_regressor.predict(input_features)[0]
    
    # Predict Temperature
    predicted_temperature = temperature_regressor.predict(input_features)[0]
    
    # Predict Rainfall
    predicted_rainfall = rainfall_regressor.predict(input_features)[0]
    
    # Suggest Power Generation Source
    suggested_power_source = 'Solar' if predicted_weather == 'Sunny' else 'Hydro'
    
    # Calculate Total Power Needed to be Produced (more than demand)
    total_power_needed = predicted_energy_demand * 1.1  
    
    # Calculate Power to be Produced by Solar and Hydro
    if suggested_power_source == 'Solar':
        power_hydro = total_power_needed / 5
        power_solar = total_power_needed - power_hydro
    else:
        power_solar = total_power_needed / 5
        power_hydro = total_power_needed - power_solar
    
    return {
        'Date':input_date,
        'Predicted_Weather': predicted_weather,
        'Predicted_Energy_Demand_GWh': predicted_energy_demand,
        'Suggested_Power_Source': suggested_power_source,
        'Total_Power_Needed_GWh': total_power_needed,
        'Power_Produced_by_Solar_GWh': power_solar,
        'Power_Produced_by_Hydro_GWh': power_hydro,
        'Predicted_Temperature_Celsius': predicted_temperature,
        'Predicted_Rainfall_mm': predicted_rainfall
    }
# Function to predict for user input date, temperature, and rainfall
def predict_for_user_input_temp_rain(input_date, input_temperature, input_rainfall):
    # Extract year and month from the input date
    input_year = input_date.year
    input_month = input_date.month
    
    # Use the previous year's data to predict
    previous_year_data = train_data[(train_data['Year'] == (input_year - 1)) & (train_data['Month'] == input_month)]
    input_features = previous_year_data[features].values.reshape(1, -1)
    
    # Predict Energy Demand
    predicted_energy_demand = xgb_regressor.predict(input_features)[0]
    
    # Suggest Power Generation Source
    suggested_power_source = 'Solar' if input_temperature > 25 and input_rainfall < 50 else 'Hydro'
    
    # Predict Weather
    predicted_weather = rf_classifier.predict(input_features)[0] if input_temperature and input_rainfall is None else 'Rainy' if input_rainfall >= 50 else 'Sunny'
    
    # Calculate Total Power Needed to be Produced (more than demand)
    total_power_needed = predicted_energy_demand * 1.1  
    
    # Calculate Power to be Produced by Solar and Hydro
    if suggested_power_source == 'Solar':
        power_hydro = total_power_needed / 5
        power_solar = total_power_needed - power_hydro
    else:
        power_solar = total_power_needed / 5
        power_hydro = total_power_needed - power_solar
    
    return {
        'Predicted_Weather': predicted_weather,
        'Predicted_Energy_Demand_GWh': predicted_energy_demand,
        'Suggested_Power_Source': suggested_power_source,
        'Total_Power_Needed_GWh': total_power_needed,
        'Power_Produced_by_Solar_GWh': power_solar,
        'Power_Produced_by_Hydro_GWh': power_hydro
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    
    if request.method == 'POST':
        user_input_date_str = request.form['input_date']
        user_input_date = pd.to_datetime(user_input_date_str, format='%Y-%m')
        prediction = predict_for_user_input(user_input_date)
    
    return render_template('index.html', prediction=prediction)
@app.route('/predict_temperature_rainfall', methods=['GET', 'POST'])
def predict_with_temperature_rainfall():
    prediction = None
    
    if request.method == 'POST':
        user_input_date_str = request.form['input_date']
        user_input_date = pd.to_datetime(user_input_date_str, format='%Y-%m')
        user_input_temperature = float(request.form['input_temperature'])
        user_input_rainfall = float(request.form['input_rainfall'])
        prediction = predict_for_user_input_temp_rain(user_input_date, user_input_temperature, user_input_rainfall)
    
    return render_template('predict_temperature_rainfall.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
