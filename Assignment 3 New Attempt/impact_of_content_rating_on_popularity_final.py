import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Create output directory for visualizations
output_dir = "output_visualizations"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load dataset
data = pd.read_csv("/content/drive/MyDrive/DV Assignment 3/Google-Playstore-Preprocessed.csv")  # Replace with your dataset

# Ensure necessary columns exist
required_columns = ["Content Rating", "Maximum Installs"]
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"Dataset must contain the following columns: {required_columns}")

# Step 2: Preprocess data
data = data.dropna(subset=["Content Rating", "Maximum Installs"])
data = data[data["Maximum Installs"] > 0]

# Define high/low installs threshold
install_threshold = data["Maximum Installs"].median()
data["Install_Category"] = np.where(data["Maximum Installs"] > install_threshold, "High", "Low")

# Encode labels for classification
label_encoder = LabelEncoder()
data["Install_Category_Encoded"] = label_encoder.fit_transform(data["Install_Category"])

# Step 3: Initial Visualization
def visualize_content_rating_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=data.groupby("Content Rating")[["Maximum Installs"]].mean().reset_index(),
        x="Content Rating",
        y="Maximum Installs",
    )
    plt.title("Average Installs by Content Rating")
    plt.ylabel("Average Maximum Installs")
    plt.xticks(rotation=45)
    file_path = os.path.join(output_dir, "average_installs_by_content_rating.png")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

visualize_content_rating_distribution(data)

# Step 4: Classification function
def train_and_evaluate_classifier(data, iteration=None):
    features = pd.get_dummies(data["Content Rating"], drop_first=True)
    target = data["Install_Category_Encoded"]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    file_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

    # Feature Importance Visualization
    feature_importance = pd.DataFrame({
        "Feature": features.columns,
        "Importance": classifier.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x="Importance", y="Feature")
    plt.title("Feature Importance for Install Prediction")
    plt.xlabel("Importance")
    plt.ylabel("Content Rating Categories")

    # Ensure unique file name for each iteration
    if iteration is None:
        file_path = os.path.join(output_dir, "initial_feature_importance.png")
    else:
        file_path = os.path.join(output_dir, f"feature_importance_feedback_loop_{iteration}.png")

    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

train_and_evaluate_classifier(data)

# Step 5: Feedback loop for merging underrepresented categories
def merge_categories(data, merge_map):
    data["Content Rating"] = data["Content Rating"].replace(merge_map)
    return data

feedback_iterations = 2
for i in range(feedback_iterations):
    print(f"Feedback Loop {i + 1}")
    
    # Visualize current distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=data.groupby("Content Rating")[["Maximum Installs"]].mean().reset_index(),
        x="Content Rating",
        y="Maximum Installs",
    )
    plt.title(f"Average Installs by Content Rating (Feedback Loop {i + 1})")
    plt.ylabel("Average Maximum Installs")
    plt.xticks(rotation=45)
    file_path = os.path.join(output_dir, f"feedback_loop_{i + 1}_average_installs.png")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

    # Display current counts of Content Rating categories
    print("Current Content Rating counts:")
    print(data["Content Rating"].value_counts())

    # User interaction: Prompt to merge categories
    print("\nEnter the categories to merge in the format 'source:target' (e.g., 'Unrated:Everyone').")
    print("Enter multiple mappings separated by commas (e.g., 'Adults only 18+:Mature, Unrated:Everyone').")
    print("Leave blank to skip merging in this iteration.")
    user_input = input("Enter your merge mappings: ").strip()

    if user_input:
        merge_map = {}
        try:
            # Parse user input into a dictionary
            for mapping in user_input.split(","):
                source, target = mapping.split(":")
                merge_map[source.strip()] = target.strip()

            # Apply the merging
            data = merge_categories(data, merge_map)
        except ValueError:
            print("Invalid input format. Please use 'source:target' pairs separated by commas.")
            continue

    # Retrain and re-evaluate
    train_and_evaluate_classifier(data, iteration=i + 1)

# Step 6: Final Visualization
def final_visualization(data):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x="Content Rating", y="Maximum Installs", showfliers=False)
    plt.title("Final Install Distribution by Content Rating")
    plt.xticks(rotation=45)
    file_path = os.path.join(output_dir, "final_install_distribution_by_content_rating.png")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

final_visualization(data)