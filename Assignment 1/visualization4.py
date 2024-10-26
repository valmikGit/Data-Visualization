import pandas as pd
import numpy as np

df = pd.read_csv("Google-Playstore-Preprocessed.csv")

# VISUALIZATION 4

# Finding the success of apps based for different content ratings based on average maximum installs
contents = df['Content Rating'].unique()
result:dict = {}
for content in contents:
    filtered_Df = df[df['Content Rating'] == content]
    installs = 0
    count = 0
    for i in range(len(filtered_Df)):
        installs += df.iloc[i]['Maximum Installs']
        count += 1
    result[content] = installs/count
for key, value in result.items():
    print(f'{key}: {value}')

import matplotlib.pyplot as plt
categories:list = []
values: list = []

for key, value in result.items():
    categories.append(key)
    values.append(value)

# Creating the bar graph
bars = plt.bar(categories, values, color='skyblue')  # Adjust color as needed

# Add value labels inside each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # x position of text
        2 * (yval / 3),  # y position of text (adjusted to be inside the bar)
        f'{yval: .2f}',  # text to display
        ha='center',  # horizontal alignment
        va='center',  # vertical alignment
        color='black',  # text color
        fontsize=8,  # font size
        rotation=90
    )

# Adding titles and labels
plt.title('Success of apps for various content ratings based on average maximum installs')
plt.xlabel('Content Rating')
plt.ylabel('Average Maximum Installs')
plt.xticks(rotation=90)

# Display the plot
plt.show()