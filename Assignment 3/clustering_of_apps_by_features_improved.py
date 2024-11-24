import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('Google-Playstore-Preprocessed.csv')

# Data Preprocessing
df['Price'] = df['Price'].replace('0', np.nan)  # Replace '0' with NaN for Price column
df['Price'] = df['Price'].apply(lambda x: x.replace('$', '') if isinstance(x, str) else x)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Fill missing values
imputer = SimpleImputer(strategy="mean")
df['Rating'] = imputer.fit_transform(df[['Rating']])

# Convert categorical variables to numerical using One-Hot Encoding
df = pd.get_dummies(df, columns=['Category', 'Content Rating', 'Currency'], drop_first=True)

# Map boolean-like columns (Free, Ad Supported, In App Purchases, Editor's Choice) to 1/0
df['Free'] = df['Free'].map({'True': 1, 'False': 0})
df['Ad Supported'] = df['Ad Supported'].map({'True': 1, 'False': 0})
df['In App Purchases'] = df['In App Purchases'].map({'True': 1, 'False': 0})
df['Editors Choice'] = df['Editors Choice'].map({'True': 1, 'False': 0})

# Select features for clustering (numerical and one-hot encoded features)
features = ['Rating', 'Rating Count', 'Minimum Installs', 'Maximum Installs', 'Price', 'Size'] + \
           [col for col in df.columns if col.startswith('Category_') or col.startswith('Content Rating_') or col.startswith('Currency_')]

X = df[features].dropna()  # Dropping any rows with missing data in selected features

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to find the optimal number of clusters
def optimal_kmeans_clusters(X_scaled):
    inertia = []
    k_range = range(1, 11)  # Trying cluster numbers from 1 to 10
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    
    # Plotting the elbow graph
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertia, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.show()

    # Suggest the optimal k (based on the "elbow" point)
    optimal_k = 3  # Default suggestion, can be adjusted based on elbow
    # Find the 'elbow' where the inertia starts to decrease at a slower rate
    if inertia[1] - inertia[0] > inertia[2] - inertia[1]:
        optimal_k = 3
    elif inertia[2] - inertia[1] > inertia[3] - inertia[2]:
        optimal_k = 4
    print(f"Suggested number of clusters: {optimal_k}")

# Call the Elbow Method
optimal_kmeans_clusters(X_scaled)

# Ask the user for the number of clusters based on the elbow plot
n_clusters = int(input("Enter the number of clusters you'd like to try (based on the elbow plot): "))

# Apply K-means clustering
def kmeans_clustering(n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    return df, kmeans

# First Clustering with user-defined number of clusters
df, kmeans_model = kmeans_clustering(n_clusters)

# Plot the initial clustering using Parallel Coordinates Plot
def plot_parallel_coordinates(df, features):
    fig = px.parallel_coordinates(df, color='Cluster', dimensions=features, title="Parallel Coordinates Plot of Clusters")
    fig.show()

# Plot the initial clustering
plot_parallel_coordinates(df, features)

# Feedback Loop 1: User interaction for the number of clusters
def feedback_loop_1():
    n_clusters = int(input("Enter the number of clusters you'd like to try: "))
    df, kmeans_model = kmeans_clustering(n_clusters)
    plot_parallel_coordinates(df, features)
    return df, kmeans_model

# Run the first feedback loop
df, kmeans_model = feedback_loop_1()

# Feedback Loop 2: User interaction to change feature weights for clustering
def feedback_loop_2():
    print("Current feature weights: Rating, Rating Count, Minimum Installs, Maximum Installs, Price, Size")
    feature_weights = {feature: float(input(f"Enter weight for {feature}: ")) for feature in features}
    weighted_features = np.multiply(X_scaled, list(feature_weights.values()))
    
    # Apply K-means clustering again with new feature weights
    kmeans = KMeans(n_clusters=kmeans_model.n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(weighted_features)
    
    plot_parallel_coordinates(df, features)
    return df, kmeans

# Run the second feedback loop
df, kmeans_model = feedback_loop_2()

# Final visualization with refined clusters after feedback
plot_parallel_coordinates(df, features)