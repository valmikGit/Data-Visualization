# google-play-store-apps-data-dv
Data visualization of the Google Play store apps' dataset.
- Link to the dataset: https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps

# Preprocessing and information we gathered before we began data visualization

## Preprocessing:
- Drop rows with null values.
- Drop rows with size or minimum android version = "Varies with device"
- Convert all sizes to the same units
- Remove android version specifications with no. of apps < 1000 (outliers)
- Delete unnecessary columns
- We checked whether the columns 'Installs' and 'Minimum Installs' convey the same information. We were correct and thus, we decided to drop the column 'Installs'.

## Categorical Variables:
- Category
- Installs
- Free
- Currency
- Content rating
- Ad supported
- In App purchases
- Editor's choice
- Minimum android

## Numerical Variables or those which are not categorical:
- App name
- App id
- rating
- rating count
- minimum installs
- maximum installs
- price
- currency
- released
- last updated

# Visualizations:
## Visualization 1 - Top 10 categories by average rating and maximum installs:
### Part A: Top 10 categories by average category:
- Found the unique entries in the column 'Category'.
- For each unique category, calculated the average rating = (summation(rating count * rating))/(summation(rating count)). Here, summation implies the sum of the mentioned quantity over all the filtered rows.
- Arranged the categories, in descending order, with respect to their average rating and found the top 10.
- Made a bar graph of Average Rating vs Categories (top 10).
