# ## Content Rating Analysis - Merged Code

# ### Step 1: Import Libraries and Load Data

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset (replace 'dataset.csv' with your actual dataset path)
df = pd.read_csv('dataset.csv')

# Overview of dataset
print("Dataset Overview:")
print(df.info())

# Check for null values in Content Rating and Category
print("\nChecking for null values in Content Rating and Category...")
print(df[['Content Rating', 'Category']].isnull().sum())

# ### Step 2: Data Preprocessing

# Handle missing values in 'Content Rating' and 'Category'
df['Content Rating'] = df['Content Rating'].fillna('Unknown')  # Fill missing content ratings with 'Unknown'
df['Category'] = df['Category'].fillna('Other')  # Fill missing categories with 'Other'

# Check unique values in Content Rating and Category
print("\nUnique Content Ratings:", df['Content Rating'].unique())
print("\nUnique Categories:", df['Category'].unique())

# ### Step 3: Initial Visualization

# Generate a stacked bar chart showing content rating distribution across categories
plt.figure(figsize=(12, 8))
category_content_rating_counts = pd.crosstab(df['Category'], df['Content Rating'])
category_content_rating_counts.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='Set3')

plt.title('Content Rating Distribution Across Categories', fontsize=16)
plt.xlabel('Category', fontsize=14)
plt.ylabel('Number of Apps', fontsize=14)
plt.xticks(rotation=90)
plt.legend(title='Content Rating', title_fontsize='13', fontsize='12')
plt.tight_layout()
plt.show()

# ### Step 4: Feedback Loop for User Interaction (Iteration 1)

print("\nFeedback Options:")
print("1. Merge content ratings categories with fewer than 10 apps.")
print("2. Merge similar content ratings (e.g., 'Everyone' and 'Everyone 10+').")
print("3. Proceed with the original content ratings.")
choice = int(input("Enter your choice (1, 2, or 3): "))

if choice == 1:
    # Merge categories with fewer than 10 apps in content rating
    content_rating_counts = df.groupby(['Category', 'Content Rating']).size().unstack().fillna(0)
    small_categories = content_rating_counts[content_rating_counts.sum(axis=1) < 10].index
    df['Content Rating'] = df['Content Rating'].replace(dict.fromkeys(small_categories, 'Other'))
    print(f"Aggregated small content rating categories into 'Other'.")
elif choice == 2:
    # Merge similar content rating categories (e.g., 'Everyone' and 'Everyone 10+')
    df['Content Rating'] = df['Content Rating'].replace({'Everyone 10+': 'Everyone', 'Mature 17+': 'Teen'})
    print(f"Aggregated similar content rating categories.")
elif choice == 3:
    print("Proceeding with the original dataset.")
else:
    print("Invalid choice. Proceeding with the original dataset.")

# Replot stacked bar chart after first iteration of feedback
plt.figure(figsize=(12, 8))
category_content_rating_counts = pd.crosstab(df['Category'], df['Content Rating'])
category_content_rating_counts.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='Set3')

plt.title('Content Rating Distribution Across Categories (After Feedback Iteration 1)', fontsize=16)
plt.xlabel('Category', fontsize=14)
plt.ylabel('Number of Apps', fontsize=14)
plt.xticks(rotation=90)
plt.legend(title='Content Rating', title_fontsize='13', fontsize='12')
plt.tight_layout()
plt.show()

# ### Step 5: Second Feedback Loop for User Interaction (Iteration 2)

print("\nFeedback Options (Iteration 2):")
print("1. Further merge small categories with fewer than 10 apps.")
print("2. Further simplify content rating categories (e.g., merge remaining groups).")
print("3. Proceed with the updated dataset.")
choice = int(input("Enter your choice (1, 2, or 3): "))

if choice == 1:
    # Further merge small categories with fewer than 10 apps
    content_rating_counts = df.groupby(['Category', 'Content Rating']).size().unstack().fillna(0)
    small_categories = content_rating_counts[content_rating_counts.sum(axis=1) < 10].index
    df['Content Rating'] = df['Content Rating'].replace(dict.fromkeys(small_categories, 'Other'))
    print(f"Further aggregated small content rating categories into 'Other'.")
elif choice == 2:
    # Further simplify content rating categories
    df['Content Rating'] = df['Content Rating'].replace({'Unknown': 'Other'})
    print(f"Further simplified content rating categories.")
elif choice == 3:
    print("Proceeding with the updated dataset.")
else:
    print("Invalid choice. Proceeding with the updated dataset.")

# Replot stacked bar chart after second iteration of feedback
plt.figure(figsize=(12, 8))
category_content_rating_counts = pd.crosstab(df['Category'], df['Content Rating'])
category_content_rating_counts.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='Set3')

plt.title('Content Rating Distribution Across Categories (After Feedback Iteration 2)', fontsize=16)
plt.xlabel('Category', fontsize=14)
plt.ylabel('Number of Apps', fontsize=14)
plt.xticks(rotation=90)
plt.legend(title='Content Rating', title_fontsize='13', fontsize='12')
plt.tight_layout()
plt.show()