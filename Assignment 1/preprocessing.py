import pandas as pd
import numpy as np

df = pd.read_csv("Google-Playstore.csv")

# Drop rows with null values
for column in df.columns:
		df.drop(labels=df.index[df[column].isna()],inplace=True)

print(df.shape)

# Drop rows with size or minimum android version = "Varies with device"
df.drop(labels=df.index[df["Size"] == "Varies with device"],inplace=True)
df.drop(labels=df.index[df["Minimum Android"] == "Varies with device"],inplace=True)

print(df.shape)

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

# Remove android version specifications with no. of apps < 1000 (outliers)
unique_versions = set(df["Minimum Android"])
version_cnts = {}
for s in unique_versions:
    version_cnts[s] = sum(df["Minimum Android"].str.count(s))

df["vcount"] = df["Minimum Android"].apply(lambda x: version_cnts[x])

df.drop(labels=df.index[df["vcount"] < 1000],inplace=True)

print(df.shape)

# Delete unnecessary columns
drop_columns = ["Developer Website", "Developer Email", "Scraped Time", "Privacy Policy", "vcount", "Developer Id"]
df.drop(drop_columns, inplace=True, axis=1)

print(df.shape)

df

df['Last Updated']

for col1, col2 in zip(df['Minimum Installs'], df['Installs']):
    cleaned_Installs = cleaned_string = ''.join(filter(str.isdigit, col2))
    if(int(col1) != int(cleaned_Installs)):
        print(col1, col2)

# This proves that the columns 'Installs' and 'Minimum installs' convey the same information. Thus, we decided to drop the column 'Installs'.

df.drop(['Installs'], inplace=True, axis=1)
df

# Convert the processed dataframe to a csv file.
df.to_csv("Google-Playstore-Preprocessed.csv")