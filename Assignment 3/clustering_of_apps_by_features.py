# import os  # Import os for folder management
# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load dataset
# df = pd.read_csv('Google-Playstore-Preprocessed.csv')

# # Select relevant numerical features for clustering
# features = ['Rating', 'Rating Count', 'Minimum Installs', 'Price']
# data = df[features].dropna()  # Drop rows with missing values for simplicity

# print("Dataset loaded and preprocessed for clustering.")

  
# # ### Step 2: Standardize Features

  
# # Standardize the features
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(data)

# print("Features standardized.")

  
# # ### Step 3: Perform Clustering

  
# # Apply K-Means clustering
# kmeans = KMeans(n_clusters=3, random_state=42)
# data['Cluster'] = kmeans.fit_predict(data_scaled)

# print("K-Means clustering applied with 3 clusters.")

  
# # ### Step 4: Visualize Clusters

  
# # Using PCA for Dimensionality Reduction

# # Reduce dimensions to 2D for visualization
# pca = PCA(n_components=2)
# data_pca = pca.fit_transform(data_scaled)
# data['PCA1'] = data_pca[:, 0]
# data['PCA2'] = data_pca[:, 1]

# # Scatter plot of clusters
# plt.figure(figsize=(10, 6))
# sns.scatterplot(
#     x='PCA1', y='PCA2', hue='Cluster', data=data, palette='viridis', s=100
# )
# plt.title('Clustering of Apps (PCA)')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend(title='Cluster')

# # Create output folder if not already created
# output_folder = "output_for_clustering_of_apps_by_features"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # Ask user for output file name and save plot
# file_name = input("Enter the name for the output visualization file (without extension): ")
# output_path = os.path.join(output_folder, f"{file_name}.png")
# plt.savefig(output_path)
# print(f"Visualization saved at {output_path}")

# # plt.show()

  
# # ### Step 5: Feedback Loop for Refinement

  
# # Focus on a Specific Cluster

# # Display cluster summary
# print("Cluster Summary:")
# print(data.groupby('Cluster').mean())

# # Ask user for feedback
# print("\nFeedback Options:")
# print("1. Focus on a specific cluster (e.g., Cluster 0).")
# print("2. Proceed without refinement.")

# # User input
# choice = int(input("Enter your choice (1 or 2): "))

# if choice == 1:
#     cluster_choice = int(input("Enter the cluster number to focus on (0, 1, or 2): "))
#     focused_data = data[data['Cluster'] == cluster_choice].reset_index(drop=True)
#     print(f"Focused on Cluster {cluster_choice}.")
# else:
#     focused_data = data
#     print("Proceeding without refinement.")

  
# # ### Step 6: Re-cluster the Focused Dataset

  
# # Reapply Clustering

# # Reapply clustering on the focused dataset
# if not focused_data.empty:
#     focused_data_scaled = scaler.fit_transform(focused_data[features])
#     focused_kmeans = KMeans(n_clusters=2, random_state=42)
#     focused_data['SubCluster'] = focused_kmeans.fit_predict(focused_data_scaled)

#     # Re-visualize clusters
#     pca_focused = PCA(n_components=2)
#     focused_pca = pca_focused.fit_transform(focused_data_scaled)
#     focused_data['PCA1'] = focused_pca[:, 0]
#     focused_data['PCA2'] = focused_pca[:, 1]

#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(
#         x='PCA1', y='PCA2', hue='SubCluster', data=focused_data, palette='coolwarm', s=100
#     )
#     plt.title(f'Re-Clustering of Focused Apps (Cluster {cluster_choice})')
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.legend(title='SubCluster')

#     # Ask user for output file name and save re-clustering plot
#     file_name = input("Enter the name for the focused cluster output file (without extension): ")
#     output_path = os.path.join(output_folder, f"{file_name}.png")
#     plt.savefig(output_path)
#     print(f"Focused cluster visualization saved at {output_path}")

#     # plt.show()
# else:
#     print("No data to re-cluster.")

import os  # Import os for folder management
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Google-Playstore-Preprocessed.csv')

# Select relevant numerical features for clustering
features = ['Rating', 'Rating Count', 'Minimum Installs', 'Price']
data = df[features].dropna()  # Drop rows with missing values for simplicity

print("Dataset loaded and preprocessed for clustering.")

  
# ### Step 2: Standardize Features

  
# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

print("Features standardized.")

  
# ### Step 3: Perform Clustering

  
# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

print("K-Means clustering applied with 3 clusters.")

  
# ### Step 4: Visualize Clusters

  
# Using PCA for Dimensionality Reduction

# Reduce dimensions to 2D for visualization
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
data['PCA1'] = data_pca[:, 0]
data['PCA2'] = data_pca[:, 1]

# Scatter plot of clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='PCA1', y='PCA2', hue='Cluster', data=data, palette='viridis', s=100
)
plt.title('Clustering of Apps (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')

# Create output folder if not already created
output_folder = "output_for_clustering_of_apps_by_features"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Ask user for output file name and save plot
file_name = input("Enter the name for the output visualization file (without extension): ")
output_path = os.path.join(output_folder, f"{file_name}.png")
plt.savefig(output_path)
print(f"Visualization saved at {output_path}")

# plt.show()

  
# ### Step 5: Feedback Loop for Refinement

  
# Focus on a Specific Cluster

# Display cluster summary
print("Cluster Summary:")
print(data.groupby('Cluster').mean())

# Ask user for feedback
print("\nFeedback Options:")
print("1. Focus on a specific cluster (e.g., Cluster 0).")
print("2. Proceed without refinement.")

# User input
choice = int(input("Enter your choice (1 or 2): "))

if choice == 1:
    cluster_choice = int(input("Enter the cluster number to focus on (0, 1, or 2): "))
    focused_data = data[data['Cluster'] == cluster_choice].reset_index(drop=True)
    print(f"Focused on Cluster {cluster_choice}.")
else:
    focused_data = data
    print("Proceeding without refinement.")

  
# ### Step 6: Re-cluster the Focused Dataset

  
# Reapply Clustering

# Reapply clustering on the focused dataset
if not focused_data.empty:
    focused_data_scaled = scaler.fit_transform(focused_data[features])
    focused_kmeans = KMeans(n_clusters=2, random_state=42)
    focused_data['SubCluster'] = focused_kmeans.fit_predict(focused_data_scaled)

    # Re-visualize clusters
    pca_focused = PCA(n_components=2)
    focused_pca = pca_focused.fit_transform(focused_data_scaled)
    focused_data['PCA1'] = focused_pca[:, 0]
    focused_data['PCA2'] = focused_pca[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='PCA1', y='PCA2', hue='SubCluster', data=focused_data, palette='coolwarm', s=100
    )
    plt.title(f'Re-Clustering of Focused Apps (Cluster {cluster_choice})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='SubCluster')

    # Ask user for output file name and save re-clustering plot
    file_name = input("Enter the name for the focused cluster output file (without extension): ")
    output_path = os.path.join(output_folder, f"{file_name}.png")
    plt.savefig(output_path)
    print(f"Focused cluster visualization saved at {output_path}")

    # plt.show()

    # Second Feedback Loop
    print("\nFeedback Options:")
    print("1. Further refine a sub-cluster (e.g., SubCluster 0).")
    print("2. Stop refinement and finalize.")

    # User input for second feedback loop
    choice_2 = int(input("Enter your choice (1 or 2): "))

    if choice_2 == 1:
        sub_cluster_choice = int(input("Enter the sub-cluster number to focus on (0 or 1): "))
        further_focused_data = focused_data[focused_data['SubCluster'] == sub_cluster_choice].reset_index(drop=True)
        print(f"Focused on SubCluster {sub_cluster_choice}.")

        if not further_focused_data.empty:
            further_scaled = scaler.fit_transform(further_focused_data[features])
            pca_final = PCA(n_components=2)
            further_pca = pca_final.fit_transform(further_scaled)
            further_focused_data['PCA1'] = further_pca[:, 0]
            further_focused_data['PCA2'] = further_pca[:, 1]

            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                x='PCA1', y='PCA2', data=further_focused_data, s=100
            )
            plt.title(f'Focused View of SubCluster {sub_cluster_choice}')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')

            # Save the second feedback visualization
            file_name = input("Enter the name for the second feedback output file (without extension): ")
            output_path = os.path.join(output_folder, f"{file_name}.png")
            plt.savefig(output_path)
            print(f"Second feedback visualization saved at {output_path}")

            # plt.show()
        else:
            print("No data to further refine in the selected sub-cluster.")
    else:
        print("Refinement process finalized.")
else:
    print("No data to re-cluster.")