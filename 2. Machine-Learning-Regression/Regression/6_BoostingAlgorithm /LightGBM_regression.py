import pandas as pd

# Load the dataset
data = pd.read_csv('insurance_pre.csv')

# Create dummies for categorical variables
data = pd.get_dummies(data, drop_first=True)

# independent and dependent variables
indipendent_variables = data.drop('charges', axis=1)
dependent_variable = data['charges']
# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(indipendent_variables, dependent_variable, test_size=0.2, random_state=42)

# Import the LightGBM Regressor
import lightgbm as lgb
# Create the LightGBM Regressor
regressor = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# Fit the regressor to the training data
regressor.fit(X_train, y_train)
# Make predictions on the test data
y_pred = regressor.predict(X_test)
# Ensure y_pred is a 1D numpy array
import numpy as np
y_pred = np.asarray(y_pred).ravel()
# Evaluate the regressor
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
# Feature importance

###################################   Mean Squared Error: 18345503.925012916
###################################   R^2 Score: 0.8818315373740427