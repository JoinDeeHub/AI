# Concept: Predicting Daily CO₂ Emissions from Car Travel Based on Distance Driven
# Description:
# This dataset models the relationship between the daily distance a person drives (in kilometers) and the resulting CO₂ emissions (in kilograms). This is a valuable sustainability metric for understanding the environmental impact of personal transportation and encouraging more sustainable commuting habits.

# Columns:

# DistanceKm (Distance driven per day in kilometers)
# CO2Kg (CO₂ emissions per day in kilograms)

import pandas as pd

# Load the dataset

dataset = pd.read_csv("car_travel_co2_emissions.csv")

#print(dataset)

# Split the dataset into (input & output) independent and dependent variables
independent = dataset[["DistanceKm"]]
dependent = dataset[["CO2Kg"]]

# print(independent)
# print(dependent)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(independent, dependent, test_size=0.3, random_state=0)
#print(X_train, X_test, Y_train, Y_test)

#model creation
from sklearn.linear_model import LinearRegression
LinearRegression = LinearRegression()
LinearRegression.fit(X_train, Y_train)

# The weight is the coefficient of the linear regression model, which represents the change in the dependent variable (CO2Kg) for a one-unit change in the independent variable (DistanceKm).
weight = LinearRegression.coef_
print("Weight: ", weight)

# The bias is the intercept of the linear regression model, which represents the value of the dependent variable (CO2Kg) when the independent variable (DistanceKm) is zero.
bias = LinearRegression.intercept_
print("Bias: ", bias)

#MODEL TRAINING

y_predict = LinearRegression.predict(X_test)
print("Predicted CO2Kg: ", y_predict)

#MODEL EVALUATION
from sklearn.metrics import r2_score
r2score = r2_score(Y_test, y_predict) 
print("R2 Score: ", r2score)
# R2 Score is nearlly 1, which indicates that the model is a good fit for the data.

#proceed to save the model
import pickle
# Save the model
good_fit_model_name = 'CO2Kg_linear_r_model.pkl'   # Save the model to disk #file_name = 'CO2Kg_linear_r_model.sav'

pickle.dump(LinearRegression, open(good_fit_model_name, 'wb'))

# Load the model
load_good_fit_model = pickle.load(open(good_fit_model_name, "rb"))

Result = load_good_fit_model.predict([[50]])
print("Predicted CO2Kg for 50 Km: ", Result)                               