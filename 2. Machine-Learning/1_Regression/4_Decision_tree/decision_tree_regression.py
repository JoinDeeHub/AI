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


# Import the Decision Tree Regressor from sklearn.tree
from sklearn.tree import DecisionTreeRegressor
# Create the Decision Tree Regressor model
regressor = DecisionTreeRegressor(criterion='poisson', splitter='random')
# Fit the model to the training data
regressor.fit(X_train, y_train)
# Predict the results
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
# Calculate the R2 score
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2}")

import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
# Visualize the Decision Tree
from sklearn import tree
#plt.ion()
tree.plot_tree(regressor, filled=True)
#plt.show()
# Save the Decision Tree as an image instead of displaying it as a plot using plt.show()
plt.figure(figsize=(90,80))
tree.plot_tree(regressor, filled=True, feature_names=list(independent_variables.columns))
plt.savefig("decision_tree.png")
print("Decision tree saved as decision_tree.png")

                 ##########################  R2 Score:  0.973325133405058
