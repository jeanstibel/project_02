# functions 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

# Get the data info
def data_info(data_df, num_fill_strategy=0, cat_fill_strategy='mode'):
    """
    Processes a dataset by analyzing it and handling missing values.

    Parameters:
    - data_df (pd.DataFrame): The DataFrame to process.
    - num_fill_strategy (int/float): Value to fill missing numerical values. Default is 0.
    - cat_fill_strategy (str): Strategy to fill missing categorical values. Options are 'mode' or a specific value. Default is 'mode'.

    Returns:
    - pd.DataFrame: The cleaned and processed DataFrame.
    """
    # Print basic information about the data
    print("=== Basic Data Information ===")
    print(f"Shape of the data: {data_df.shape}")
    print("\nColumns in the dataset:")
    print(data_df.columns)
    print("\nData types:")
    print(data_df.dtypes)
    
    # Print missing values information
    print("\n=== Missing Values Information ===")
    print("Percentage of missing values per column:")
    print(data_df.isnull().mean())
    print("\nCount of missing values per column:")
    print(data_df.isna().sum())
    
    # Handle missing values
    print("\n=== Handling Missing Values ===")
    for column in data_df.columns:
        if data_df[column].isnull().any():
            if data_df[column].dtype == 'object':  # Categorical column
                if cat_fill_strategy == 'mode':
                    fill_value = data_df[column].mode()[0]
                else:
                    fill_value = cat_fill_strategy
                data_df[column].fillna(fill_value, inplace=True)
                print(f"Filled missing values in '{column}' with '{fill_value}' (categorical).")
            else:  # Numerical column
                data_df[column].fillna(num_fill_strategy, inplace=True)
                print(f"Filled missing values in '{column}' with '{num_fill_strategy}' (numerical).")
    
    # Recheck for missing values
    print("\n=== Missing Values After Handling ===")
    print("Count of missing values per column after handling:")
    print(data_df.isna().sum())
    
    # Print statistical overview
    print("\n=== Statistical Overview ===")
    print(data_df.describe(include='all'))
    
    return data_df

# Clean and process the raw cryptocurrency data
def clean_and_process_data(df):
    """
    Clean and process the raw cryptocurrency data.

    Parameters:
    df (pd.DataFrame): The raw cryptocurrency data.

    Returns:
    pd.DataFrame: The cleaned and processed data.
    """
    # Convert 'timestamp' to datetime and extract date
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    # Fix 'timestamp' format to HH:MM:SS
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%H:%M:%S')
    
    # Rename column 'timestamp' to 'open_time'
    df.rename(columns={'timestamp': 'open_time'}, inplace=True)
    
    # Fix 'close_time' format to HH:MM:SS
    df['close_time'] = pd.to_datetime(df['close_time']).dt.strftime('%H:%M:%S')
    
    # Reorder columns
    df = df[['crypto_id', 'date', 'open_time', 'close_time', 'open', 'close', 'high', 'low', 'volume', 'number_of_trades']]
    
    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Group by 'date' and 'crypto_id' and aggregate data
    df = df.groupby(['date', 'crypto_id']).agg({
        'open': 'first',
        'close': 'last',
        'high': 'max',
        'low': 'min',
        'volume': 'sum',
        'number_of_trades': 'sum'
    }).reset_index()
    
    # Sort by 'crypto_id' and 'date'
    df = df.sort_values(by=['crypto_id', 'date'])
    
    # Calculate additional features
    df['diff_oc'] = df['close'] - df['open']
    df['return'] = ((df['close'] - df['open']) / df['open']) * 100
    df['volatility'] = ((df['high'] - df['low']) / df['open']) * 100
    df['vol_change'] = df.groupby('crypto_id')['volume'].pct_change() * 100
    df['ma_5'] = df.groupby('crypto_id')['close'].transform(lambda x: x.rolling(window=5).mean())
    df['ma_10'] = df.groupby('crypto_id')['close'].transform(lambda x: x.rolling(window=10).mean())
    
    return df

# Fill missing values based on data type
def fill_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'float64' or df[column].dtype == 'int64':
            df[column].fillna(0, inplace=True)
        elif df[column].dtype == 'object':
            df[column].fillna('unknown', inplace=True)
        elif df[column].dtype == 'datetime64[ns]':
            df[column].fillna(pd.Timestamp('1900-01-01'), inplace=True)
    return df

# Scale the data for classification
def classification_models(X_train_scaled, y_train, X_test_scaled, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
        'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=5, random_state=42),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42),
        'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'AdaBoostClassifier': AdaBoostClassifier(n_estimators=100, random_state=42),
        'BaggingClassifier': BaggingClassifier(n_estimators=100, random_state=42),
        'SVC': SVC(kernel='linear', probability=True, random_state=42),
    }

    classification_results = {}
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        classification_results[model_name] = {
            'Accuracy': accuracy,
            'Classification Report': class_report
        }

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    # Display classification results
    classification_results_df = pd.DataFrame({model: {'Accuracy': res['Accuracy']} for model, res in classification_results.items()})
    return classification_results_df

