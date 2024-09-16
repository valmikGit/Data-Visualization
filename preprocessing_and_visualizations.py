# %%
import pandas as pd
import numpy as np

# %%
df = pd.read_csv("Google-Playstore.csv")

# %%
# Drop rows with null values
for column in df.columns:
		df.drop(labels=df.index[df[column].isna()],inplace=True)

print(df.shape)

# %%
# Drop rows with size or minimum android version = "Varies with device"
df.drop(labels=df.index[df["Size"] == "Varies with device"],inplace=True)
df.drop(labels=df.index[df["Minimum Android"] == "Varies with device"],inplace=True)

print(df.shape)

# %%
# Convert all sizes to the same units
def normalize_size(size):
    nsize = float(size[:-1].replace(",", ""))
    if size[-1] == "G":
        return nsize*1024
    elif size[-1] == "k":
        return nsize/1024
    else:
        return nsize

df['Size'] = df['Size'].apply(normalize_size)

# %%
# Remove android version specifications with no. of apps < 1000 (outliers)
unique_versions = set(df["Minimum Android"])
version_cnts = {}
for s in unique_versions:
    version_cnts[s] = sum(df["Minimum Android"].str.count(s))

df["vcount"] = df["Minimum Android"].apply(lambda x: version_cnts[x])

df.drop(labels=df.index[df["vcount"] < 1000],inplace=True)

print(df.shape)

# %%
# Delete unnecessary columns
drop_columns = ["Developer Website", "Developer Email", "Scraped Time", "Privacy Policy", "vcount", "Developer Id"]
df.drop(drop_columns, inplace=True, axis=1)

print(df.shape)

# %%
df

# %%
df['Last Updated']

# %%
for col1, col2 in zip(df['Minimum Installs'], df['Installs']):
    cleaned_Installs = cleaned_string = ''.join(filter(str.isdigit, col2))
    if(int(col1) != int(cleaned_Installs)):
        print(col1, col2)

# This proves that the columns 'Installs' and 'Minimum installs' convey the same information. Thus, we decided to drop the column 'Installs'.

# %%
df.drop(['Installs'], inplace=True, axis=1)
df

# %%
# Convert the processed dataframe to a csv file.
df.to_csv("Google-Playstore-Preprocessed.csv")

# VISUALIZATION 1
# %%
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

# %%
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

# VISUALIZATION 2

# %%
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

# %%
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

# VISUALIZATION 3
# %%
df['Content Rating'].unique()

# %%
# Finding the success of apps based on their content rating by calculating average rating.
contents = df['Content Rating'].unique()
result:dict = {}
for content in contents:
    filtered_Df = df[df['Content Rating'] == content]
    total = 0
    count = 0
    for i in range(len(filtered_Df)):
        total += (df.iloc[i]['Rating Count'] * df.iloc[i]['Rating'])
        count += df.iloc[i]['Rating Count']
    result[content] = total/count
for key, value in result.items():
    print(f'{key}: {value}')

# %%
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
        yval / 2,  # y position of text (adjusted to be inside the bar)
        f'{yval}',  # text to display
        ha='center',  # horizontal alignment
        va='center',  # vertical alignment
        color='black',  # text color
        fontsize=12,  # font size
        rotation=90
    )

# Adding titles and labels
plt.title('Success of apps for various content ratings based on average rating')
plt.xlabel('Content Rating')
plt.ylabel('Average Rating')
plt.xticks(rotation=90)

# Display the plot
plt.show()

# VISUALIZATION 4

# %%
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

# %%
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


