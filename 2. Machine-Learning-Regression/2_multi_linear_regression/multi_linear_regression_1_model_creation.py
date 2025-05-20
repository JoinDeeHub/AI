import pandas as pd

# Load the dataset
data = pd.read_csv("50_Startups.csv")
# Print the dataset
#print(data)


#Create dummies for categorical variables - (Nominal variables) will be using one hot encoding
data = pd.get_dummies(data, drop_first=True)
#print(data)

#To check the columns
#print(data.columns)

# Input output split
independent_variable = data[['R&D Spend', 'Administration', 'Marketing Spend','State_Florida', 'State_New York']]

dependent_variable = data[['Profit']]

# Print the independent and dependent variables
# print(independent_variable)
# print(dependent_variable)

# Train test split    
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(independent_variable, dependent_variable, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#fit the model
regressor.fit(X_train, Y_train)

#Weight and Bias
weight = regressor.coef_
print("Weight: ", weight)

bias = regressor.intercept_
print("Bias: ", bias)

y_predict = regressor.predict(X_test)
print("Predicted Profit: ", y_predict)

#Model Evaluation
from sklearn.metrics import r2_score
r2_score = r2_score(Y_test, y_predict)
print("R2 Score: ", r2_score)


# Save the model
import pickle
good_fit_model_name = 'profit_multi_linear_r_model.pkl'   #file_name = 'profit_multi_linear_r_model.sav'
pickle.dump(regressor, open(good_fit_model_name, 'wb'))

# Load the model
load_good_fit_model = pickle.load(open(good_fit_model_name, "rb"))
result = load_good_fit_model.predict([[160000, 130000, 140000, 0, 1]])
print("Predicted Profit for 160000, 130000, 140000, 0, 1: ", result)