import pickle

good_fit_model_name = 'profit_multi_linear_r_model.pkl'   #file_name = 'profit_multi_linear_r_model.sav'

# Load the model
load_good_fit_model = pickle.load(open(good_fit_model_name, "rb"))
R_D_Spend = float(input("Enter R&D Spend to Predict Profit: "))
Administration = float(input("Enter Administration to Predict Profit: "))
Marketing_Spend = float(input("Enter Marketing Spend to Predict Profit: "))
State_Florida = int(input("Enter State Florida to Predict Profit: "))
State_New_York = int(input("Enter State New York to Predict Profit: "))
result = load_good_fit_model.predict([[R_D_Spend, Administration, Marketing_Spend, State_Florida, State_New_York]])
print("Predicted Profit for R&D Spend, Administration, Marketing Spend, State Florida, State New York: ", result)
# The model is trained using the training data (X_train and Y_train) to learn the relationship between the independent and dependent variables.
