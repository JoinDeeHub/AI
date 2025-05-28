# Simple Linear Regression Model Creation
# Importing the required libraries
import pandas as pf

# Load the dataset
#data import -> collection or preprocessing
data = pf.read_csv("Salary_Data.csv")
# print(data)


#input output split
independenrt_variable = data[["YearsExperience"]]

dependent_variable = data[["Salary"]]

# print(independenrt_variable)
# print(dependent_variable)

#train test split
from sklearn.model_selection import train_test_split

X_tarin, X_test, Y_train, Y_test = train_test_split(independenrt_variable, dependent_variable, test_size=0.2, random_state=0)
#print(X_tarin, X_test, Y_train, Y_test)


#MODEL CREATION
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_tarin, Y_train)

# The weight is the coefficient of the linear regression model, which represents the change in the dependent variable (Salary) for a one-unit change in the independent variable (YearsExperience).
# The bias is the intercept of the linear regression model, which represents the value of the dependent variable (Salary) when the independent variable (YearsExperience) is zero.
#MODEL TRAINING
# The model is trained using the training data (X_train and Y_train) to learn the relationship between the independent and dependent variables.
# The model is then used to make predictions on the test data (X_test) to evaluate its performance.
weight = regressor.coef_
print("Weight: ", weight)

bias = regressor.intercept_
print("Bias: ", bias)

#MODEL PREDICTION
# Predicting the test set results
Y_predict = regressor.predict(X_test)
print("Predicted Salary: ", Y_predict)

#MODEL EVALUATION
from sklearn.metrics import r2_score 
r2_score = r2_score(Y_test, Y_predict)
print("R2 Score: ", r2_score)


import pickle
# Save the model
file_name = 'salary_linear_r_model.pkl'   # Save the model to disk #file_name = 'salary_linear_r_model.sav' 
pickle.dump(regressor, open(file_name, 'wb'))

# Load the model
load_model = pickle.load(open(file_name, 'rb'))
result = load_model.predict([[10]])
print("Predicted Salary for 10 years of experience: ", result)