import pickle

good_fit_model_filename = "random_forest_regression_model.pkl"

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
from sklearn.ensemble import RandomForestRegressor

# Check if the loaded model is a Random Forest
if isinstance(load_good_fit_model, RandomForestRegressor):
	single_tree = load_good_fit_model.estimators_[0]
else:
	raise TypeError("The loaded model is not a Random Forest Regressor.")
tree.plot_tree(single_tree, filled=True)
#plt.show()
# Save the plot as an image
plt.figure(figsize=(90,80))
tree.plot_tree(single_tree, filled=True, feature_names=list(load_good_fit_model.feature_importances_))
# Save the plot as an image
plt.savefig("random_forest_final.png")