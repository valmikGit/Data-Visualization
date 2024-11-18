  
# ## Content Rating Analysis Part 2

  
# ### Step 1: Initial Setup and Data Loading

 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset (replace 'dataset.csv' with your actual dataset path)
df = pd.read_csv('dataset.csv')

# Check for null values in Content Rating and Category
print("Dataset Overview:")
print(df.info())


  
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


  
# ### Step 4: Feedback Loop for User Interaction

 
# Prompt user for interaction to aggregate categories based on content ratings
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


  
# ### Step 5: Re-Visualization After Feedback

 
# Replot stacked bar chart after user feedback
plt.figure(figsize=(12, 8))
category_content_rating_counts = pd.crosstab(df['Category'], df['Content Rating'])
category_content_rating_counts.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='Set3')

plt.title('Content Rating Distribution Across Categories (After Feedback)', fontsize=16)
plt.xlabel('Category', fontsize=14)
plt.ylabel('Number of Apps', fontsize=14)
plt.xticks(rotation=90)
plt.legend(title='Content Rating', title_fontsize='13', fontsize='12')
plt.tight_layout()
plt.show()



