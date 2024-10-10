import pandas as pd
import os

def load_datasets(folder_path):
    folder_path = "./archive" #hardcoded file path to ./archive
    data_dict = {}  # creating empty dictionary for dataframe
    csv_files = os.listdir(folder_path) # going to archive folder for csv files
    for file in csv_files:
        if file.endswith(".csv"): #for every file in archive that ends with .csv
            full_path = os.path.join(folder_path, file) #combine the path with the file name
            df = pd.read_csv(full_path) #reading the csv file into the dataframe
            data_dict[file] = df # add the csv filename to the empty dictionary
    return data_dict
    

# explores a dataset
# Step 2: This function should print out basic info about a selected dataset, such as its first 5 rows, column names, data types, etc.
def explore_dataset(df):
    print("First 5 rows: \n")
    print(df.head())

    print("Column names: \n")
    print(df.columns)

    print("Data types: \n")
    print(df.dtypes)

    print("Missing values: \n")
    print(df.isnull().sum())

    print("Summary stats: \n")
    print(df.describe())

    print('\n')

data_dict = load_datasets('./archive')
explore_dataset(data_dict["races.csv"])