# Concept
# Predicting Monthly Electricity Bill for Households

# Description
# This dataset models the relationship between various household factors and the resulting monthly electricity bill. The goal is to predict the monthly electricity bill (in USD) based on:

# SquareFootage: The size of the house or apartment in square feet. Larger homes typically consume more electricity due to increased space for lighting, heating/cooling, and appliances.
# NumOccupants: The number of people living in the household. More occupants usually lead to higher energy consumption due to increased use of appliances, lighting, and electronics.
# HeatingType: The primary type of heating system used in the household. This is a nominal categorical variable with possible values:
#   Electric: The household uses electric heating, which can significantly increase electricity usage, especially in colder months.
#   Gas: The household uses gas heating, which generally results in lower electricity consumption for heating purposes.
#   None: The household has no central heating system, or uses other methods, potentially leading to lower or highly variable electricity bills.
# AvgTemp: The average outside temperature for the billing month (in Celsius). Lower temperatures might increase heating needs, while higher temperatures may increase cooling needs, both impacting electricity usage.
# MonthlyBill: The target variable—total monthly electricity bill in USD.

import pandas as pd

# Load the dataset
dataset = pd.read_csv("household_electricity_with_category.csv")
# Print the dataset
#print(dataset)

# Create dummies for categorical variables - (Nominal variables) will be using one hot encoding
data = pd.get_dummies(dataset, drop_first=False)
print(data)