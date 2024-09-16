import pandas as pd
import numpy as np

df = pd.read_csv("Google-Playstore-Preprocessed.csv")

# VISUALIZATION 2

# Finding the top 10 categories by average maximum installs
unique_Categories = df['Category'].unique()
result:dict = {}
for category in unique_Categories:
    filtered_Df = df[df['Category'] == category]
    installs = 0
    count = 0
    for i in range(len(filtered_Df)):
        installs += df.iloc[i]['Maximum Installs']
        count += 1
    result[category] = installs/count
top_10 = sorted(result.items(), key=lambda x: x[1], reverse=True)[:10]
print(top_10)

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
plt.title('Top 10 categories of apps based on average maximum installs')
plt.xlabel('Categories')
plt.ylabel('Average Maximum installs')
plt.xticks(rotation=90)

# Display the plot
plt.show()