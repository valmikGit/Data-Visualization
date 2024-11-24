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
from sklearn.model_selection import GridSearchCV
import plotly.express as px
import os

# Create the output directory if it doesn't exist
if not os.path.exists('output_for_enhanced_analysis'):
    os.makedirs('output_for_enhanced_analysis')

# Load the dataset
df = pd.read_csv(r'/content/drive/MyDrive/Assignment 3 New Attempt/Google-Playstore-Preprocessed.csv')

# Data Preprocessing
# Handle missing or invalid values
df['Price'] = df['Price'].replace('0', np.nan)  # Replace '0' with NaN for Price column
df['Price'] = df['Price'].apply(lambda x: x.replace('$', '') if isinstance(x, str) else x)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

imputer = SimpleImputer(strategy="mean")
df['Rating'] = imputer.fit_transform(df[['Rating']])
df['Size'] = df['Size'].fillna(df['Size'].median())  # Replace with non-inplace method

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
plt.savefig("output_for_enhanced_analysis/correlation_matrix.png")
plt.close()

# Distribution Plots
for col in ['Price', 'Rating', 'Size']:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.savefig(f"output_for_enhanced_analysis/distribution_{col}.png")
    plt.close()

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

# Hyperparameter Tuning for Random Forest
def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    
    return grid_search.best_estimator_

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
    plt.savefig("output_for_enhanced_analysis/confusion_matrix_tuned.png")
    plt.close()

    return accuracy

accuracy = evaluate_model(best_rf_model, X_test, y_test)

# Interactive Visualization with Plotly
def plot_dynamic_visualization(df):
    fig = px.scatter(df, x='Minimum Installs', y='Rating', color='Category_Tools', 
                     size='Size', hover_data=['Price'], title='App Ratings vs Minimum Installs')
    fig.show()

plot_dynamic_visualization(df)