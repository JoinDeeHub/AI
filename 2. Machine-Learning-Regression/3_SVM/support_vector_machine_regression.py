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

####################### Inorder to improve the performance of the model, lets used "Standardization Method", - we will scale the features using StandardScaler
# from sklearn.preprocessing import StandardScaler
# # Create a StandardScaler object
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

###################### METHOD 1 ######################   ______________________

# Import the Support Vector Regressor SVR from sklearn.svm, using the "linear" kernel
from sklearn.svm import SVR
# Create the SVR model
regressor = SVR(kernel='linear')
# Fit the model to the training data
regressor.fit(X_train, y_train)

                ##########################   R2 Score:  0.8742528560882514 
                
                
# ####################### METHOD 2 ##################      O

# Import the Support Vector Regressor SVR from sklearn.svm, using "Radial Basis Function" (RBF) kernel with specific hyperparameters:
    #   - `C=1000`: Regularization parameter. A higher value of `C` reduces the margin of tolerance for misclassified points, leading to a more complex model.
    #   - `gamma='scale'`: Kernel coefficient for the RBF kernel. The 'scale' option uses `1 / (n_features * X.var())` as the value of gamma, which is a good default for most datasets.
    #   - `epsilon=0.1`: Specifies the epsilon-tube within which no penalty is given for errors in the training data. Smaller values make the model more sensitive to small deviations.

# from sklearn.svm import SVR
# # Create the SVR model
# regressor = SVR(kernel='rbf', C=1000, gamma='scale', epsilon=.1)
# # Fit the model to the training data
# regressor.fit(X_train, y_train)

                ##########################  R2 Score:  0.04311992341899473
                
                
###################### METHOD 3 ###################       U

# # Import the Support Vector Regressor SVR from sklearn.svm, using the "poly" kernel
# from sklearn.svm import SVR
# # Create the SVR model
# regressor = SVR(kernel='poly')
# # Fit the model to the training data
# regressor.fit(X_train, y_train)

                ##########################   R2 Score:  -0.15101677287478932  
                

###################### METHOD 4 ##################        ~

# Import the Support Vector Regressor SVR from sklearn.svm, using the "sigmoid" kernel
# from sklearn.svm import SVR
# # Create the SVR model
# regressor = SVR(kernel='sigmoid')
# # Fit the model to the training data
# regressor.fit(X_train, y_train)

                ##########################   R2 Score:  -0.1598967128719493    
                
                
###################### METHOD 5 #################        Must be a square matrix

# # Import the Support Vector Regressor SVR from sklearn.svm, using the "precomputed" kernel
# from sklearn.svm import SVR
# # Create the SVR model
# regressor = SVR(kernel='precomputed')
# # Fit the model to the training data
# regressor.fit(X_train, y_train)

                ##########################   ValueError: Precomputed matrix must be a square matrix. Input is a 40x6 matrix.                         

# weight = regressor.coef_
# print("Weight: ", weight)

# bias = regressor.intercept_
# print("Bias: ", bias)
regressor.n_support_
# Print the number of support vectors for each class
print("Number of support vectors for each class: ", regressor.n_support_)

regressor.support_
# Print the indices of the support vectors
print("Indices of support vectors: ", regressor.support_)

regressor.support_vectors_
#print("Support vectors: ", regressor.support_vectors_)

# Predict the results
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
# Evaluate the model
r2 = r2_score(y_test, y_pred)
print("R2 Score: ", r2)

#The Support Vector Regressor SVR, using the "linear" kernel we are getting the best R2 score of 0.8742528560882514, which indicates a good fit for the model.
##########################   R2 Score:  0.8742528560882514  This is the best model without standardization gives R2(-0.1572)

#CONCLUSTION :
                    # FOR THIS DATASET: ----- SVM ------ dosen't work well with non-linear data, and the linear kernel is the best choice for this dataset.
                    
                    # Lets see using other alogorithms like Decision Tree, Random Forest, and XGBoost to see if we can get a better R2 score.