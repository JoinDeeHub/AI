class univariate():
    #This logic works perfectly to separate the columns into qualitative and quantitative categories.
    # now lets create a function to reuse this logic
    def qualitative_quantitative(self, dataset):
        quantitative = []
        qualitative = []
        
        for column in dataset.columns:
            if dataset[column].dtype == 'object':
                qualitative.append(column)
            else:
                quantitative.append(column)
        
        return qualitative, quantitative
    # Now we can use the function to separate columns in our dataset
    # Example usage:
    # uni = univariate()
    # qualitative_columns, quantitative_columns = uni.qualitative_quantitative(dataset)
    # print(f"Qualitative Columns: {qualitative_columns}\n")
    # print(f"Quantitative Columns: {quantitative_columns}")