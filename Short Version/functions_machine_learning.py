# functions_machine_learning 
import pandas as pd
import numpy as np
from sklearn.base import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

# Import train_test_split
def split_data(features_df, features, target, test_size=0.2, random_state=42):
    # Split the data into features (X) and target (y)
    X = features_df[features]
    y = features_df[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

# Scale the data for regression
def train_and_evaluate_models(X_train, y_train, models):
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        mse = mean_squared_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        results.append({"Model": name, "MSE": mse, "R2 Score": r2})
    
    results_df = pd.DataFrame(results)
    return results_df





















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
