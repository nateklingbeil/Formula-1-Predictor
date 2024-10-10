import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# datasets are the individual .csv's
# function to load CSV files from ./archive and return a dictionary.
# The key is the file name, and the value is the DataFrame
def load_datasets(folder_path):
    folder_path = "./archive"  # Hardcoded path to ./archive
    data_dict = {}  # Create an empty dictionary to store DataFrames
    csv_files = os.listdir(folder_path)  # List all files in the folder
    for file in csv_files:
        if file.endswith(".csv"):  # Select only CSV files
            full_path = os.path.join(folder_path, file)  # Full path to file
            df = pd.read_csv(full_path)  # Read CSV into DataFrame
            data_dict[file] = df  # Add to dictionary
    return data_dict



# function explores basic info about a selected dataset, such as its first 5 rows, column names, data types, etc.
def explore_dataset(df):
    print("\n---First 5 rows---")
    print(df.head())  # Display first 5 rows
    
    print("\n---Column names---")
    print(df.columns)  # List column names
    
    print("\n---Data types---")
    print(df.dtypes)  # Display column data types
    
    print("\n---Missing values---")
    print(df.isnull().sum())  # Show missing values per column
    
    print("\n---Summary stats---")
    print(df.describe())  # Summary statistics for numerical columns
    print('\n')

# function lists all available datasets (i.e., print the keys from the `data_dict`).
def list_available_datasets(data_dict):
    print("\n---Available datasets---\n")
    for dataset in data_dict.keys():  # Iterate over each file name in the dictionary
        print(dataset)  # Print the file name

# select a dataset from the available list and explore it
def select_and_explore_dataset(data_dict):
    list_available_datasets(data_dict) #lists all datasets in data_dict
    ds_choice = input("\nSelect a dataset to explore: ")
    if ds_choice in data_dict.keys(): #checks if users choice is in data_dict
        print(f"\n---Exploring {ds_choice}---")
        explore_dataset(data_dict[ds_choice]) #if the users choice exists, pass the dataframe to explore_dataset
    else:
        print("---The dataset you selected does not exist.---")


# Merge two datasets (`results.csv` and `races.csv`) based on common column like `raceId`.
def merge_results_races(data_dict):
    results_df = data_dict["results.csv"]  # Loads race results data from the dict (results.csv)
    races_df = data_dict["races.csv"]  # Load the race details data from the dictionary (races.csv)

    # Merge results_df and races_df on the common column 'raceId' using a left join
    # 'left join' means all rows from results_df will be kept, and matching rows from races_df will be added
    merged_race_data = pd.merge(results_df, races_df, on='raceId', how="left")  
  
    return merged_race_data  # Return the merged DataFrame

# handle missing data in the merged dataset
# Step 5: Replace missing values (`\N`) and remove unnecessary columns, and clean any rows where essential data is missing.
def clean_data(df_combined):
    pass  # <-- Your code here

# Convert data types in relevant columns
# Step 6: Convert specific columns (like `position`, `grid`, and `points`) to numeric data types for analysis.
def convert_data_types(df_cleaned):
    pass  # <-- Your code here

# Selecting relevant columns for analysis
# Step 7: Select specific columns (e.g., 'grid', 'position') from the cleaned dataset to use in analysis or visualizations.
def select_features(df_cleaned):
    pass  # <-- Your code here

# Heatmap to show the relationship between grid position and final position
# Step 8: Create a heatmap to visualize the relationship between grid position and final position.
def explore_data_heatmap(df_selected):
    pass  # <-- Your code here

# MAIN PROGRAM
# if __name__ == "__main__":

# Load all datasets from the ./archive folder
data_dict = load_datasets('./archive')

# List all datasets
# list_available_datasets(data_dict)

# Select and explore a dataset
# select_and_explore_dataset(data_dict)

# # Explore a specific dataset
# explore_dataset(data_dict["races.csv"])  # Change 'races.csv' to whatever

# Merge results.csv and races.csv, then explore the merged dataset
merged_data = merge_results_races(data_dict)
explore_dataset(merged_data)
print(merged_data.head(10))


