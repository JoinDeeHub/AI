import pandas as pd

# Load the dataset
dataset = pd.read_csv("50_Startups.csv")
# Create dummies for categorical variables - (Nominal variables) will be using one hot encoding
dataset = pd.get_dummies(dataset, drop_first=False)
# To check the columns

# print(dataset.columns)
independent_variables = dataset[['R&D Spend', 'Administration', 'Marketing Spend', 'State_California', 'State_Florida', 'State_New York']]

dependent_variables = dataset[['Profit']]

from sklearn.model_selection import train_test_split
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(independent_variables, dependent_variables, test_size=0.2, random_state=0)

# Import the Random Forest Regressor from sklearn.ensemble
###### We know under Random Forest Regressor, have sklearn.ensemble, sklearn.bagging, sklearn.boostrapping, sklearn.randomfeaturesselection
from sklearn.ensemble import RandomForestRegressor
# Create the Random Forest Regressor model
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
# Fit the model to the training data
regressor.fit(X_train, y_train)
# Predict the results
y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score
# Calculate the R2 score
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2}")

import pickle
# Save the model to a file
good_fit_model_filename = "random_forest_regression_model.pkl"
pickle.dump(regressor, open(good_fit_model_filename, 'wb'))
# Load the model from the file
load_good_fit_model = pickle.load(open("random_forest_regression_model.pkl", "rb"))
# Ensure the input has all 6 features (including the missing one-hot encoded column for 'State_New York')
result = load_good_fit_model.predict([[160000, 130000, 140000, 0, 1, 0]])
print("Predicted Profit for 160000, 130000, 140000, 0, 1, 0: ", result)

# Visualize the Random Forest
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# Visualize the Random Forest
from sklearn import tree
#plt.ion()
# Extract a single decision tree from the Random Forest
single_tree = regressor.estimators_[0]
tree.plot_tree(single_tree, filled=True)
#plt.show()
plt.savefig("random_forest.png")
print("Random Forest saved as random_forest.png")
