import pickle

# Load the model
load_model = pickle.load(open("salary_linear_r_model.pkl", 'rb'))
result = load_model.predict([[10]])
print("Predicted Salary for 10 years of experience: ", result)
# The above code is a simple linear regression model that predicts salary based on years of experience.
# It uses the scikit-learn library for model creation and evaluation, and the pickle library for saving and loading the model.
# The model is trained on a dataset of salary and years of experience, and the prediction is made for 10 years of experience.
# The model is saved to a file named 'salary_linear_r_model.pkl' and loaded back for making predictions.
# The predicted salary for 10 years of experience is printed to the console.