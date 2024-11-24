 
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import GridSearchCV
# import plotly.express as px
# import os

# # # GPU Setup (Optional)
# # import tensorflow as tf
# # print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# # Output Directory in Google Drive
# output_dir = 'output_for_enhanced_analysis'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Load the dataset
# df = pd.read_csv('Google-Playstore-Preprocessed.csv')

# # Data Preprocessing
# # Handle missing or invalid values
# df['Price'] = df['Price'].replace('0', np.nan)  # Replace '0' with NaN for Price column
# df['Price'] = df['Price'].apply(lambda x: x.replace('$', '') if isinstance(x, str) else x)
# df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# imputer = SimpleImputer(strategy="mean")
# df['Rating'] = imputer.fit_transform(df[['Rating']])
# df['Size'] = df['Size'].fillna(df['Size'].median())

# # Convert problematic columns to ensure numeric-only correlation matrix
# non_numeric_columns = df.select_dtypes(exclude=['number']).columns
# df[non_numeric_columns] = df[non_numeric_columns].apply(lambda x: pd.factorize(x)[0])

# # Encode categorical variables
# categorical_columns = ['Category', 'Content Rating', 'Ad Supported', 'In App Purchases', 'Editors Choice']
# df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# # Feature Engineering
# # Create derived features
# df['Installs Range'] = df['Maximum Installs'] - df['Minimum Installs']
# df['Price Category'] = pd.cut(df['Price'].fillna(0), bins=[0, 1, 10, 50, np.inf], labels=['Free', 'Low', 'Medium', 'High'])
# df = pd.get_dummies(df, columns=['Price Category'], drop_first=True)

# # Normalize numerical features
# scaler = StandardScaler()
# num_features = ['Rating Count', 'Price', 'Size', 'Installs Range']
# df[num_features] = scaler.fit_transform(df[num_features])

# # Define binary target variable: 'Success'
# df['Success'] = ((df['Rating'] > 4) & (df['Minimum Installs'] > 1000000)).astype(int)

# # Features and target
# all_features = [col for col in df.columns if col not in ['Success', 'Rating', 'Minimum Installs', 'Maximum Installs']]
# X = df[all_features]
# y = df['Success']

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Initial Visualization
# # Correlation Matrix Heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.savefig(f"{output_dir}/correlation_matrix.png")
# plt.close()

# # Distribution Plots
# for col in ['Price', 'Rating', 'Size']:
#     plt.figure(figsize=(8, 6))
#     sns.histplot(df[col].dropna(), kde=True, bins=30)
#     plt.title(f'Distribution of {col}')
#     plt.savefig(f"{output_dir}/distribution_{col}.png")
#     plt.close()

 
# # Model Training and Comparison
# def train_and_compare_models(X_train, X_test, y_train, y_test):
#     models = {
#         'Random Forest': RandomForestClassifier(random_state=42),
#         'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
#     }

#     results = {}
#     for name, model in models.items():
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         results[name] = accuracy

#         print(f"{name} Accuracy: {accuracy:.2f}")
#         print(classification_report(y_test, y_pred))

#     return results

# results = train_and_compare_models(X_train, X_test, y_train, y_test)

 
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV
# import numpy as np

# # Hyperparameter tuning using RandomizedSearchCV
# def tune_hyperparameters(X_train, y_train):
#     param_dist = {
#         'n_estimators': [50, 100, 150],
#         'max_depth': [10, 20, None],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4]
#     }

#     # Using RandomizedSearchCV for faster search
#     randomized_search = RandomizedSearchCV(
#         estimator=RandomForestClassifier(random_state=42),
#         param_distributions=param_dist,
#         n_iter=20,  # Limit the number of iterations for quicker search
#         cv=2,  # Use 2-fold cross-validation
#         scoring='accuracy',
#         n_jobs=-1,  # Use all available cores
#         random_state=42,
#         verbose=1  # Print progress
#     )

#     randomized_search.fit(X_train, y_train)

#     print("Best Parameters:", randomized_search.best_params_)
#     print("Best Score:", randomized_search.best_score_)

#     return randomized_search.best_estimator_

# # Example usage
# best_rf_model = tune_hyperparameters(X_train, y_train)

 
# # Evaluate the tuned model
# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     cm = confusion_matrix(y_test, y_pred)

#     print(f"Model Accuracy: {accuracy:.2f}")
#     print("Confusion Matrix:")
#     print(cm)

#     # Save confusion matrix plot
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Success', 'Success'], yticklabels=['Not Success', 'Success'])
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.savefig(f"{output_dir}/confusion_matrix_tuned.png")
#     plt.close()

#     return accuracy

# accuracy = evaluate_model(best_rf_model, X_test, y_test)

 
# # Importing necessary libraries
# import pandas as pd
# import plotly.express as px

# # Define the function for dynamic visualization
# def plot_dynamic_visualization(df):
#     # Check if the required columns exist in the dataframe
#     required_columns = ['Minimum Installs', 'Rating', 'Size', 'Price']
#     category_column = None  # Placeholder for identifying the correct category column

#     # Search for a valid category column dynamically
#     for col in df.columns:
#         if "Category" in col:
#             category_column = col
#             break

#     # Ensure the required columns and category column are available
#     if not all(col in df.columns for col in required_columns):
#         missing_cols = [col for col in required_columns if col not in df.columns]
#         raise ValueError(f"Missing required columns: {missing_cols}")
#     if not category_column:
#         raise ValueError("No category column found in the dataframe.")

#     # Ensure 'Size' column has non-negative values
#     df['Size'] = df['Size'].apply(lambda x: abs(x))  # Take absolute value of 'Size'

#     # Plot using plotly express
#     fig = px.scatter(df,
#                      x='Minimum Installs',
#                      y='Rating',
#                      color=category_column,
#                      size='Size',
#                      hover_data=['Price'],
#                      title='App Ratings vs Minimum Installs')
#     fig.show()

# # Example usage
# # Assuming `df` is already loaded as your dataframe
# plot_dynamic_visualization(df)


# UPPER CODE IS WORKING.


# GOD ATTEMPT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import plotly.express as px
import os

# # GPU Setup (Optional)
# import tensorflow as tf
# print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Output Directory in Google Drive
output_dir = 'output_for_enhanced_analysis'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the dataset
df = pd.read_csv('Google-Playstore-Preprocessed.csv')

# Data Preprocessing
# Handle missing or invalid values
df['Price'] = df['Price'].replace('0', np.nan)  # Replace '0' with NaN for Price column
df['Price'] = df['Price'].apply(lambda x: x.replace('$', '') if isinstance(x, str) else x)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

imputer = SimpleImputer(strategy="mean")
df['Rating'] = imputer.fit_transform(df[['Rating']])
df['Size'] = df['Size'].fillna(df['Size'].median())

# Convert problematic columns to ensure numeric-only correlation matrix
non_numeric_columns = df.select_dtypes(exclude=['number']).columns
df[non_numeric_columns] = df[non_numeric_columns].apply(lambda x: pd.factorize(x)[0])

# Encode categorical variables
categorical_columns = ['Category', 'Content Rating', 'Ad Supported', 'In App Purchases', 'Editors Choice']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Feature Engineering
# Create derived features
df['Installs Range'] = df['Maximum Installs'] - df['Minimum Installs']
df['Price Category'] = pd.cut(df['Price'].fillna(0), bins=[0, 1, 10, 50, np.inf], labels=['Free', 'Low', 'Medium', 'High'])
df = pd.get_dummies(df, columns=['Price Category'], drop_first=True)

# Normalize numerical features
scaler = StandardScaler()
num_features = ['Rating Count', 'Price', 'Size', 'Installs Range']
df[num_features] = scaler.fit_transform(df[num_features])

# Define binary target variable: 'Success'
df['Success'] = ((df['Rating'] > 4) & (df['Minimum Installs'] > 1000000)).astype(int)

# Features and target
all_features = [col for col in df.columns if col not in ['Success', 'Rating', 'Minimum Installs', 'Maximum Installs']]
X = df[all_features]
y = df['Success']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initial Visualization
# Correlation Matrix Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig(f"{output_dir}/correlation_matrix.png")
plt.close()

# Distribution Plots
for col in ['Price', 'Rating', 'Size']:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.savefig(f"{output_dir}/distribution_{col}.png")
    plt.close()

# Interactive Heatmap Customization
def interactive_heatmap(df):
    print("\nInteractive Heatmap Tool")
    print("Available Columns:")
    print(list(df.columns))
    selected_columns = input("Enter columns for heatmap (comma-separated): ").split(',')

    # Check if selected columns exist
    selected_columns = [col.strip() for col in selected_columns if col.strip() in df.columns]
    if not selected_columns:
        print("No valid columns selected. Using all numeric columns.")
        selected_columns = df.select_dtypes(include=['number']).columns

    corr_matrix = df[selected_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Customized Correlation Matrix')
    plt.show()

interactive_heatmap(df)

# Model Training and Comparison
def train_and_compare_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

        print(f"{name} Accuracy: {accuracy:.2f}")
        print(classification_report(y_test, y_pred))

    return results

results = train_and_compare_models(X_train, X_test, y_train, y_test)

# Feature Selection for Model Training
def feature_selection_train(df, y):
    print("\nFeature Selection Tool")
    print("Available Features:")
    print(list(df.columns))
    selected_features = input("Enter features to include in model training (comma-separated): ").split(',')

    # Validate and use selected features
    selected_features = [feat.strip() for feat in selected_features if feat.strip() in df.columns]
    if not selected_features:
        print("No valid features selected. Using all available features.")
        selected_features = df.columns

    X_selected = df[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    print(f"Training model with features: {selected_features}")
    return train_and_compare_models(X_train, X_test, y_train, y_test)

results_selected_features = feature_selection_train(X, y)

# Hyperparameter tuning using RandomizedSearchCV
def tune_hyperparameters(X_train, y_train):
    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Using RandomizedSearchCV for faster search
    randomized_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=20,  # Limit the number of iterations for quicker search
        cv=2,  # Use 2-fold cross-validation
        scoring='accuracy',
        n_jobs=-1,  # Use all available cores
        random_state=42,
        verbose=1  # Print progress
    )

    randomized_search.fit(X_train, y_train)

    print("Best Parameters:", randomized_search.best_params_)
    print("Best Score:", randomized_search.best_score_)

    return randomized_search.best_estimator_

# Example usage
best_rf_model = tune_hyperparameters(X_train, y_train)

# Evaluate the tuned model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(cm)

    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Success', 'Success'], yticklabels=['Not Success', 'Success'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"{output_dir}/confusion_matrix_tuned.png")
    plt.close()

    return accuracy

accuracy = evaluate_model(best_rf_model, X_test, y_test)

# Dynamic Visualization
def plot_dynamic_visualization(df):
    print("\nDynamic Visualization Tool")
    required_columns = ['Minimum Installs', 'Rating', 'Size', 'Price']
    category_column = None
    for col in df.columns:
        if "Category" in col:
            category_column = col
            break

    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing_cols}")
    if not category_column:
        raise ValueError("No category column found in the dataframe.")

    df['Size'] = df['Size'].apply(lambda x: abs(x))
    fig = px.scatter(df,
                     x='Minimum Installs',
                     y='Rating',
                     color=category_column,
                     size='Size',
                     hover_data=['Price'],
                     title='App Ratings vs Minimum Installs')
    fig.show()

plot_dynamic_visualization(df)