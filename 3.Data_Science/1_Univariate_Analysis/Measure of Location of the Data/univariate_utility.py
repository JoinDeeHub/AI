import pandas as pd
import numpy as np
# This module contains utility functions for univariate analysis.
# It includes functions to separate qualitative and quantitative columns,
# create frequency tables, and perform descriptive statistics on quantitative data.

class univariate():
    #This logic works perfectly to separate the columns into qualitative and quantitative categories.
    # now lets create a function to reuse this logic
    def qualitative_quantitative(self, dataset):
        quantitative_columns = []
        qualitative_columns = []

        for column in dataset.columns:
            if dataset[column].dtype == 'object':
                qualitative_columns.append(column)
            else:
                quantitative_columns.append(column)

        return qualitative_columns, quantitative_columns
    # Now we can use the function to separate columns in our dataset
    # Example usage:
    # uni = univariate()
    # qualitative_columns, quantitative_columns = uni.qualitative_quantitative(dataset)
    # print(f"Qualitative Columns: {qualitative_columns}\n")
    # print(f"Quantitative Columns: {quantitative_columns}")
    
    def frequency_table(column, dataset):    
        Frequency_table = pd.DataFrame(columns=["unique_values", "Frequency", "Relative_Frequency", "Cumulative_Frequency"])
        
        Frequency_table["unique_values"] = dataset[column].value_counts().index
        Frequency_table["Frequency"] = dataset[column].value_counts().values
        Frequency_table["Relative_Frequency"] = (Frequency_table["Frequency"] / 103)
        Frequency_table["Cumulative_Frequency"] = Frequency_table["Relative_Frequency"].cumsum()
        
        return Frequency_table

    def Univariate(dataset,quantitative_columns):
        descriptive_stats=pd.DataFrame(index=["Mean","Median","Mode","Q1:25%","Q2:50%",
                                "Q3:75%","99%","Q4:100%","IQR","1.5rule","Lesser_outlier","Greater_outlier",
                                        "Min","Max","kurtosis","skew","Var","Std"],columns=quantitative_columns)
        for column in quantitative_columns:
            descriptive_stats[column]["Mean"]=dataset[column].mean()
            descriptive_stats[column]["Median"]=dataset[column].median()
            descriptive_stats[column]["Mode"]=dataset[column].mode()[0]
            descriptive_stats[column]["Q1:25%"]=dataset.describe()[column]["25%"]
            descriptive_stats[column]["Q2:50%"]=dataset.describe()[column]["50%"]
            descriptive_stats[column]["Q3:75%"]=dataset.describe()[column]["75%"]
            descriptive_stats[column]["99%"]=np.percentile(dataset[column],99)
            descriptive_stats[column]["Q4:100%"]=dataset.describe()[column]["max"]
            descriptive_stats[column]["IQR"]=descriptive_stats[column]["Q3:75%"]-descriptive_stats[column]["Q1:25%"]
            descriptive_stats[column]["1.5rule"]=1.5*descriptive_stats[column]["IQR"]
            descriptive_stats[column]["Lesser_outlier"]=descriptive_stats[column]["Q1:25%"]-descriptive_stats[column]["1.5rule"]
            descriptive_stats[column]["Greater_outlier"]=descriptive_stats[column]["Q3:75%"]+descriptive_stats[column]["1.5rule"]
            descriptive_stats[column]["Min"]=dataset[column].min()
            descriptive_stats[column]["Max"]=dataset[column].max()
            descriptive_stats[column]["kurtosis"]=dataset[column].kurtosis()
            descriptive_stats[column]["skew"]=dataset[column].skew()
            descriptive_stats[column]["Var"]=dataset[column].var()
            descriptive_stats[column]["Std"]=dataset[column].std()
        return descriptive_stats
    
    def outlier_detection(dataset,quantitative_columns):
        outliers = pd.DataFrame(columns=["Column", "Lesser Outlier", "Greater Outlier"])
        for column in quantitative_columns:
            lesser_outlier = dataset[dataset[column] < dataset[column].quantile(0.25) - 1.5 * (dataset[column].quantile(0.75) - dataset[column].quantile(0.25))]
            greater_outlier = dataset[dataset[column] > dataset[column].quantile(0.75) + 1.5 * (dataset[column].quantile(0.75) - dataset[column].quantile(0.25))]
            outliers = outliers.append({"Column": column, "Lesser Outlier": lesser_outlier.shape[0], "Greater Outlier": greater_outlier.shape[0]}, ignore_index=True)
        return outliers
    # Example usage:
    # uni = univariate()
    # outliers = uni.outlier_detection(dataset, quantitative_columns)
    # print(outliers)       
    
# Note: The above code assumes that the dataset is a pandas DataFrame and that the columns are numeric for quantitative analysis.
    # Function to remove outliers using the IQR method
    # This function removes outliers from the dataset based on the IQR method.
    def remove_outliers_iqr(dataset, quantitative_columns):
        for column in quantitative_columns:
            Q1 = dataset[column].quantile(0.25)
            Q3 = dataset[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            dataset = dataset[(dataset[column] >= lower_bound) & (dataset[column] <= upper_bound)]
        return dataset
    # Example usage:
    # uni = univariate()
    # cleaned_dataset = uni.remove_outliers_iqr(dataset, quantitative_columns)
    # print(cleaned_dataset)