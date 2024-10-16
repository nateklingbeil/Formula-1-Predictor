import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# datasets are the individual .csv's
# function to load CSV files from ./archive and return a dictionary.
# The key is the file name, and the value is the DataFrame
def load_datasets(folder_path):
    folder_path = "./archive"  # Hardcoded path to ./archive for now
    data_dict = {}  # Create an empty dictionary to store DataFrames
    csv_files = os.listdir(folder_path)  # List all files in the folder
    for file in csv_files:
        if file.endswith(".csv"):  # Select only CSV files
            full_path = os.path.join(folder_path, file)  # Full path to file
            df = pd.read_csv(full_path)  # Read CSV into DataFrame
            data_dict[file] = df  # Add to dictionary
    return data_dict

# function explores basic info about a selected dataset
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
    print("\n")

# function lists all available datasets (i.e., print the keys from the data_dict).
def list_available_datasets(data_dict):
    print("\n---Available datasets---\n")
    for dataset in data_dict.keys():  # Iterate over each file name in the dictionary
        print(dataset)  # Print the file name

# Merge two datasets (results.csv and races.csv) based on common column like raceId.
def merge_results_races(data_dict):
    results_df = data_dict["results.csv"]  # Loads race results data from the dict (results.csv)
    races_df = data_dict["races.csv"]  # Load the race details data from the dictionary (races.csv)
    merged_race_data = pd.merge(results_df, races_df, on="raceId", how="left")
    return merged_race_data  # Return the merged DataFrame

# handle missing data in the merged dataset
def clean_data(dataframe):
    # Replace placeholder values '\N' with NaN
    dataframe.replace({"\\N": pd.NA}, inplace=True)
    columns_to_remove = [
        "resultId", "time_x", "fastestLapTime", "fastestLapSpeed", "number", "rank", "url", 
        "fp1_date", "fp1_time", "fp2_date", "fp2_time", "fp3_date", "fp3_time", "quali_date", 
        "quali_time", "sprint_date", "sprint_time", "date", "time_y"
    ]
    dataframe.drop(columns=columns_to_remove, errors="ignore", inplace=True)  # Dropping columns that aren't useful

    essential_columns = ["raceId", "driverId", "grid", "laps", "year", "positionOrder"]
    dataframe.dropna(subset=essential_columns, inplace=True)  # Dropping rows with missing data in the essential columns
    return dataframe

# Convert data types in relevant columns
def convert_data_types(df_cleaned):
    cols_to_convert = ["grid", "laps", "points", "qualifying_position"]
    for col in cols_to_convert:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")
    return df_cleaned

# Selecting relevant columns for analysis
def select_features(df_cleaned):
    driver_group = df_cleaned.groupby("driverId")  # Groups all rows with the same driverId
    df_cleaned["driver_rolling_avg_points"] = (
        driver_group["points"].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    constructor_group = df_cleaned.groupby("constructorId")
    df_cleaned["constructor_rolling_avg_points"] = (
        constructor_group["points"].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    # Keeping 'positionOrder', 'raceId', and 'qualifying_position' for analysis
    columns_to_keep = [
        "driverId", "constructorId", "grid", "laps", "points", "year", "circuitId", "raceId", 
        "driver_rolling_avg_points", "constructor_rolling_avg_points", "positionOrder", "qualifying_position"
    ]

    df_selected = df_cleaned[columns_to_keep]
    return df_selected

# Step to add qualifying position to the dataset
def add_qualifying_position(data_dict, merged_data):
    # Load qualifying dataset
    qualifying_df = data_dict["qualifying.csv"]

    # Select necessary columns for merging
    qualifying_columns = ["raceId", "driverId", "position"]
    qualifying_df = qualifying_df[qualifying_columns]
    qualifying_df.rename(columns={"position": "qualifying_position"}, inplace=True)

    # Merge qualifying data with the merged race data
    merged_data_with_qualifying = pd.merge(merged_data, qualifying_df, on=["raceId", "driverId"], how="left")
    return merged_data_with_qualifying

if __name__ == "__main__":
    # Load datasets
    data_dict = load_datasets("./archive")

    # Clean and merge data
    merged_data = merge_results_races(data_dict)
    merged_data_with_qualifying = add_qualifying_position(data_dict, merged_data)
    cleaned_data = clean_data(merged_data_with_qualifying)
    converted_data = convert_data_types(cleaned_data)

    # Select relevant features for analysis
    selected_data = select_features(converted_data)

    # Encode categorical data using LabelEncoder
    label_encoder_driver = LabelEncoder()
    selected_data.loc[:, 'driverId'] = label_encoder_driver.fit_transform(selected_data['driverId'])
    
    label_encoder_constructor = LabelEncoder()
    selected_data.loc[:, 'constructorId'] = label_encoder_constructor.fit_transform(selected_data['constructorId'])

    # Scale the features
    features_df = selected_data[[
        'driver_rolling_avg_points', 'constructor_rolling_avg_points', 'grid', 'circuitId', 'qualifying_position'
    ]]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)

    # Define target
    target = selected_data['positionOrder']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.3, random_state=42)

    # Print confirmation
    print("\nData split completed: training and testing sets are ready.")

    # Step 1: Create an instance of RandomForestClassifier with hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

    # Step 2: Train the model
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Step 3: Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Step 4: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the optimized Random Forest model: {accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))