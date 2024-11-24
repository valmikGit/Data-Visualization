import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('Google-Playstore-Preprocessed.csv')

# Data Preprocessing
df['Price'] = df['Price'].replace('0', np.nan)  # Replace '0' with NaN for Price column
df['Price'] = df['Price'].apply(lambda x: x.replace('$', '') if isinstance(x, str) else x)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Fill missing values for Rating and Rating Count
imputer = SimpleImputer(strategy="mean")
df['Rating'] = imputer.fit_transform(df[['Rating']])

# Convert categorical variables to numerical using LabelEncoder for some columns
le = LabelEncoder()
df['Content Rating'] = le.fit_transform(df['Content Rating'])

# Map boolean-like columns (Free, Ad Supported, In App Purchases, Editor's Choice) to 1/0
df['Free'] = df['Free'].map({'True': 1, 'False': 0})
df['Ad Supported'] = df['Ad Supported'].map({'True': 1, 'False': 0})
df['In App Purchases'] = df['In App Purchases'].map({'True': 1, 'False': 0})
df['Editors Choice'] = df['Editors Choice'].map({'True': 1, 'False': 0})

# Fill missing 'Size' by filling NaN with the median size (for simplicity)
df['Size'].fillna(df['Size'].median(), inplace=True)

# Select features for statistical analysis and classification
features = ['Rating', 'Rating Count', 'Minimum Installs', 'Maximum Installs', 'Price', 'Size', 'Content Rating']

# Bar Chart: Average Ratings per Content Rating Group
def plot_avg_ratings_by_content_rating(df):
    avg_ratings = df.groupby('Content Rating')['Rating'].mean().sort_values(ascending=False)
    avg_ratings.plot(kind='bar', figsize=(10, 6))
    plt.title('Average Rating by Content Rating')
    plt.xlabel('Content Rating')
    plt.ylabel('Average Rating')
    plt.show()

# Heatmap: Success metrics (Installs, Ratings) across different Content Ratings
def plot_success_heatmap(df):
    content_rating_group = df.groupby('Content Rating')[['Rating', 'Minimum Installs']].mean()
    plt.figure(figsize=(8, 6))
    sns.heatmap(content_rating_group, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Success Metrics (Rating & Installs) by Content Rating')
    plt.show()

# Initial Visualization and Inference
plot_avg_ratings_by_content_rating(df)
plot_success_heatmap(df)

# Inference: From the heatmap, we can see that content rating types like 'Everyone' and 'Teen' have a higher average rating, which could be important for model prediction.

# Apply Classification Models to Predict App Success
# Create a binary target variable: 'Success' (1 if rating > 4 and installs > 1000000, else 0)
df['Success'] = ((df['Rating'] > 4) & (df['Minimum Installs'] > 1000000)).astype(int)

# Features for Classification
X = df[['Rating', 'Rating Count', 'Minimum Installs', 'Maximum Installs', 'Price', 'Size', 'Content Rating']]
y = df['Success']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest Classifier
def train_classification_model(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    
    # Accuracy and Confusion Matrix
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Displaying results
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    
    return rf, accuracy, cm

# Train the initial classification model
rf_model, accuracy, cm = train_classification_model(X_train, X_test, y_train, y_test)

# Feedback Loop 1: User interaction to adjust model parameters based on initial analysis
def feedback_loop_1():
    n_estimators = int(input("Enter the number of estimators for the Random Forest Classifier: "))
    max_depth = int(input("Enter the maximum depth for the Random Forest Classifier: "))
    
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    
    # Accuracy and Confusion Matrix
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Refined Model Accuracy: {accuracy * 100:.2f}%")
    print("Refined Confusion Matrix:")
    print(cm)
    
    return rf, accuracy, cm

# Run the first feedback loop
rf_model, accuracy, cm = feedback_loop_1()

# Feedback Loop 2: User interaction to include or exclude features for the model
def feedback_loop_2():
    print("Current features: Rating, Rating Count, Minimum Installs, Maximum Installs, Price, Size, Content Rating")
    include_rating_count = input("Would you like to include 'Rating Count' as a feature? (yes/no): ").lower()
    
    if include_rating_count == 'no':
        X_new = X.drop(columns=['Rating Count'])
    else:
        X_new = X
    
    # Re-split the data
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)
    
    # Train Random Forest Classifier with new features
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    
    # Accuracy and Confusion Matrix
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Refined Model with New Features Accuracy: {accuracy * 100:.2f}%")
    print("Refined Confusion Matrix:")
    print(cm)
    
    return rf, accuracy, cm

# Run the second feedback loop
rf_model, accuracy, cm = feedback_loop_2()

# Final visualizations after feedback
plot_avg_ratings_by_content_rating(df)
plot_success_heatmap(df)

# Add dynamic Plotly visualizations for user interaction
def plot_dynamic_visualization(df):
    fig = px.scatter(df, x='Minimum Installs', y='Rating', color='Content Rating', 
                     size='Size', hover_data=['Price'])
    fig.update_layout(title='App Ratings vs Minimum Installs by Content Rating',
                      xaxis_title='Minimum Installs',
                      yaxis_title='Rating')
    fig.show()

# Interactive Plotly visualization for deeper insight after user feedback
plot_dynamic_visualization(df)