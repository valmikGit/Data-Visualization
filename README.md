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
## Visualization 1 - Top 10 categories by average rating and average maximum installs:
### Part A: Top 10 categories by average rating:
- Found the unique entries in the column 'Category'.
- For each unique category, calculated the average rating = (summation(rating count * rating))/(summation(rating count)). Here, summation implies the sum of the mentioned quantity over all the filtered rows.
- Arranged the categories, in descending order, with respect to their average rating and found the top 10.
- Made a bar graph of Average Rating vs Categories (top 10).
- Our hypothesis: The number 1 category based on average rating would be Shopping. However, we realized that Parenting is the category which is on top.
- Inference: The top 10 categories based on average rating (from number 1 to number 10): Parenting, Health and fitness, Books and reference, Art and Design, Weather, Shoppping, Word, Casino, Music and Audio, Photography.
- Marks and Channels: Mark used is Line. Channel used is Length/Position. I chose this combination of mark and channel because I think, by using this combination, it is very easy to interpret the information given by the visualization.
- Type of variables used: The Y-axis has average rating which is numerical derived out of the numerical variables 'Rating Count' and 'Rating'. The X-axis has the categorical variable 'Category'.
### Part B: Top 10 categories by average maximum installs:
- Found the unique entries in the column 'Category'.
- For each unique category, calculated the maximum installs = (summation(maximum installs))/(number of filtered rows). Here, summation implies the sum of the mentioned quantity over all the filtered rows.
- Arranged the categories, in descending order, with respect to their average maximum installs and found the top 10.
- Made a bar graph of Average Maximum Installs vs Categories (top 10).
- Our hypothesis: The number 1 category based on average maximum installs would be Music and Audio. However, we realized that Comics is the category which is on top.
- Inference: The top 10 categories based on average rating (from number 1 to number 10): Comics, Education, Business, Personalization, Music and Audio, Lifestyle, Books and References, Health and Fitness, Travel and Local, Entertainment.
- Marks and Channels: Mark used is Line. Channel used is Length/Position. I chose this combination of mark and channel because I think, by using this combination, it is very easy to interpret the information given by the visualization.
- Type of variables used: The Y-axis has average maxiumum installs which is numerical derived out of the numerical variable 'Maximum Installs'. The X-axis has the categorical variable 'Category'.
- I have used average maximum installs instead of maximum installs to find the typical/most expected value of 'Maximum Installs'.

## Visualization 2 - Finding success of apps based on their content rating by calculating average rating and average maximum installs:
### Part A: Success of apps based on their content rating by calculating average rating:
- Found the unique entries in the column 'Content Rating'.
- For each unique content rating, calculated the average rating = (summation(rating count * rating))/(summation(rating count)). Here, summation implies the sum of the mentioned quantity over all the filtered rows.
- Made a bar graph of Average Rating vs Content Rating.
- Our hypothesis: The number 1 content rating based on average rating would be Everyone. Our hypothesis was corrrect.
- Inference: The order of the content rating, based on average rating, from topmost to bottommost is Everyone, Teen, Everyone 10+, Mature 17+, Adults only 18+ and Unrated.
- Marks and Channels: Mark used is Line. Channel used is Length/Position. I chose this combination of mark and channel because I think, by using this combination, it is very easy to interpret the information given by the visualization.
- Type of variables used: The Y-axis has average rating which is numerical derived out of the numerical variables 'Rating Count' and 'Rating'. The X-axis has the categorical variable 'Content Rating'.
### Part A: Success of apps based on their content rating by calculating average maximum installs:
- Found the unique entries in the column 'Content Rating'.
- For each unique content rating, calculated the maximum installs = (summation(maximum installs))/(number of filtered rows). Here, summation implies the sum of the mentioned quantity over all the filtered rows.
- Made a bar graph of Average Maximum Installs vs Content Rating.
- Our hypothesis: The number 1 content rating based on average rating would be Everyone. Our hypothesis was corrrect.
- Inference: The order of the content rating, based on average rating, from topmost to bottommost is Everyone, Teen, Everyone 10+, Mature 17+, Adults only 18+ and Unrated.
- Marks and Channels: Mark used is Line. Channel used is Length/Position. I chose this combination of mark and channel because I think, by using this combination, it is very easy to interpret the information given by the visualization.
- Type of variables used: The Y-axis has average maxiumum installs which is numerical derived out of the numerical variable 'Maximum Installs'. The X-axis has the categorical variable 'Content Rating'.
- I have used average maximum installs instead of maximum installs to find the typical/most expected value of 'Maximum Installs'.
