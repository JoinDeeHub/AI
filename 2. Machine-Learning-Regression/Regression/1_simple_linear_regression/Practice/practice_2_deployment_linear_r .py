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


import pickle
# Save the model
good_fit_model_name = 'CO2Kg_linear_r_model.pkl'   #file_name = 'CO2Kg_linear_r_model.sav'

# Load the model
load_good_fit_model = pickle.load(open(good_fit_model_name, "rb"))

km = float(input("Enter km to Predict CO2Kg: "))
Result = load_good_fit_model.predict([[km]])
print("Predicted CO2Kg for 50 Km: ", Result)                               