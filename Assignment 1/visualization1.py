
import pandas as pd
import numpy as np

df = pd.read_csv("Google-Playstore-Preprocessed.csv")

# VISUALIZATION 1
# Finding the top 10 categories by rating
unique_Categories = df['Category'].unique()
result:dict = {}
for category in unique_Categories:
    filtered_Df = df[df['Category'] == category]
    total = 0
    count = 0
    for i in range(len(filtered_Df)):
        total += (filtered_Df.iloc[i]['Rating Count'] * filtered_Df.iloc[i]['Rating'])
        count += filtered_Df.iloc[i]['Rating Count']
    result[category] = total/count
top_10 = sorted(result.items(), key=lambda x: x[1], reverse=True)[:10]

import matplotlib.pyplot as plt
categories:list = []
values: list = []

for key, value in top_10:
    categories.append(key)
    values.append(value)

# Creating the bar graph
bars = plt.bar(categories, values, color='skyblue')  # Adjust color as needed

# Add value labels inside each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # x position of text
        yval / 2,  # y position of text (adjusted to be inside the bar)
        f'{yval}',  # text to display
        ha='center',  # horizontal alignment
        va='center',  # vertical alignment
        color='black',  # text color
        fontsize=12,  # font size
        rotation=90
    )

# Adding titles and labels
plt.title('Top 10 categories of apps based on average rating')
plt.xlabel('Categories')
plt.ylabel('Averag rating')
plt.xticks(rotation=90)

# Display the plot
plt.show()