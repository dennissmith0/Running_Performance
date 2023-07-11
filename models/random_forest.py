# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/dennissmith0/Running_Performance/main/prep/training_data.csv')

# Defining helper functions that are used in the data wrangling process

def replace_nan_average_speed(df):
    df['Average Speed'] = df.apply(lambda row: row['Distance.1'] / row['Moving Time'] if pd.isna(row['Average Speed']) else row['Average Speed'], axis=1)
    return df

def remove_columns_with_prefix(df, prefix):
    columns_to_remove = [column for column in df.columns if column.startswith(prefix)]
    df.drop(columns=columns_to_remove, inplace=True)
    return df

def drop_columns_with_extra_decimals(df, suffix):
    columns_with_extra_decimals = [column for column in df.columns if column.endswith(suffix)]
    df.drop(columns=columns_with_extra_decimals, inplace=True)
    return df

def drop_columns_with_null(df):
    columns_with_null = df.columns[df.isnull().sum() >= (.50 * len(df))]
    df.drop(columns=columns_with_null, inplace=True)
    return df

def remove_columns_with_few_values(df, threshold):
    columns_to_remove = [column for column in df.columns if df[column].count() < threshold]
    df.drop(columns=columns_to_remove, inplace=True)
    return df

def impute_average_heart_rate(df):
    average_hr = df['Average Heart Rate'].mean()

    # Generate random values within the range of average_hr ± 10
    random_values = np.random.uniform(average_hr - 10, average_hr + 10, size=df['Average Heart Rate'].isnull().sum())

    # Replace missing values with the generated random values
    df.loc[df['Average Heart Rate'].isnull(), 'Average Heart Rate'] = random_values

    return df

def impute_nan_with_average(df):
    for column in df.columns:
        if df[column].isnull().any():
            average = df[column].mean()
            df[column].fillna(average, inplace=True)
    return df

# Defining the main wrangle function that combines all preprocessing steps
def wrangle(df):
    # Convert the "Activity Date" column to datetime
    df['Activity Date'] = pd.to_datetime(df['Activity Date'])

    # Sort data by date just to be sure
    df = df.sort_values('Activity Date')

    # Calculate days between activities and create new column that serves as measure for 'Rest Days'.
    df['Days Between Activity'] = df['Activity Date'].diff().dt.days

    # First value = NaN, replace with 0.
    if pd.isna(df.loc[0, 'Days Between Activity']):
        df.loc[0, 'Days Between Activity'] = 0.0

    # Now set as datetime index
    df.set_index('Activity Date', inplace=True)

    # For now, remove all activities that are not a Run
    df = df[df['Activity Type'].isin(['Run'])]
    df.drop(columns=['Activity Type'], inplace=True)

    # Call helper functions
    df = replace_nan_average_speed(df)
    df = remove_columns_with_prefix(df, '<span')
    df = drop_columns_with_extra_decimals(df, '.1')
    df = drop_columns_with_null(df)
    df = remove_columns_with_few_values(df, 50)
    df = impute_average_heart_rate(df)

    # Drop unnecessary columns
    df.drop(columns=['Elapsed Time', 'Activity ID', 'Activity Name', 'Media', 'Commute', 'From Upload', 'Filename', 'Athlete Weight',
                     'Activity Gear', 'Number of Runs', 'Prefer Perceived Exertion',
                     'Average Temperature', 'Elevation Loss', 'Gear', 'Grade Adjusted Distance'], inplace=True, errors='ignore')

    # Impute NaN values with the average value of each column
    df = impute_nan_with_average(df)

    return df

df = wrangle(df)

def calculate_training_session_intensity(df):
  # Firstly, it's advisable to normalize these features so that they are on a similar scale. This can be done using Min-Max normalization which scales the features to be between 0 and 1.
  features_to_normalize = ['Distance', 'Relative Effort', 'Elevation Gain', 'Moving Time']

  for feature in features_to_normalize:
      df[feature + "_norm"] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())

  # Create the 'Training Session Intensity' as a weighted sum of these features
  # Here we are giving equal weight (0.25) to each of the four features, but you could adjust these weights based on what you think contributes more to the intensity of a training session.
  # For example, if you think 'Relative Effort' is a more important factor, you might give it a higher weight.
  df['Training Session Intensity'] = (df['Distance_norm'] * 0.25
                                      + df['Relative Effort_norm'] * 0.25
                                      + df['Elevation Gain_norm'] * 0.25
                                      + df['Moving Time_norm'] * 0.25)

  # Drop the normalized columns, as they were just intermediates for this calculation:
  df.drop([feature + "_norm" for feature in features_to_normalize], axis=1, inplace=True)

  return df

def calculate_cumulative_load(df, window_size=7):
  df['Cumulative Load'] = df['Training Session Intensity'].rolling(window=window_size).sum()

        # Replace the NaN values for the first (window_size - 1) rows
  for i in range(window_size - 1):
    df.loc[df.index[i], 'Cumulative Load'] = df.loc[df.index[:i+1], 'Training Session Intensity'].sum()

  return df

def calculate_performance_score(df, weight_cumulative_load=0.4, weight_rest_days=0.2, weight_intensity=0.4):
  df['Performance Score'] = (weight_cumulative_load * df['Cumulative Load'] +
                                  weight_rest_days * df['Days Between Activity'] +
                                  weight_intensity * df['Training Session Intensity'])
  return df

def engineer(df):

  # Convert all necessary columns to numeric type
  columns_to_convert = ['Distance', 'Relative Effort', 'Elevation Gain', 'Moving Time', ]
                          #'Cumulative Load', 'Days Between Activity', 'Training Session Intensity']
  for column in columns_to_convert:
    df[column] = pd.to_numeric(df[column], errors='coerce')

  # Calculate Training Session Intensity
  df = calculate_training_session_intensity(df)

  # Calculate Cumulative Load
  df = calculate_cumulative_load(df)

  # Calculate Performance Score
  df = calculate_performance_score(df)

  return df

df = engineer(df)

# Split the data into features X and target y
X = df.drop('Performance Score', axis=1)
y = df['Performance Score']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a model using RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

print(f'MSE: {mse}, MAE: {mae}, R^2: {r2}, EVS: {evs}')

# Get feature importance
feature_importance = rf.feature_importances_

# Match importance with feature names
features = list(zip(X.columns, feature_importance))

# Print them out
for feature in features:
    print(f"Feature: {feature[0]}, Importance: {feature[1]}")