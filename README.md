# Assignment 1
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
- Average rating and average maximum installs, in our opinion, are excellent indicators to judge a category of apps. They can be used to judge how successful a particular category is. We mentioned the top 10 because of the large number of categories of apps. This will give an insight into the users' choice and what they generally prefer.
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
- Average rating and average maximum installs, in our opinion, are excellent indicators to judge the effect of content rating on the success of apps. This will provide an insight into how the users think and how they respond depending on the content rating of an app.
### Part A: Success of apps based on their content rating by calculating average rating:
- Found the unique entries in the column 'Content Rating'.
- For each unique content rating, calculated the average rating = (summation(rating count * rating))/(summation(rating count)). Here, summation implies the sum of the mentioned quantity over all the filtered rows.
- Made a bar graph of Average Rating vs Content Rating.
- Our hypothesis: The number 1 content rating based on average rating would be Everyone. Our hypothesis was corrrect.
- Inference: The order of the content rating, based on average rating, from topmost to bottommost is Everyone, Teen, Everyone 10+, Mature 17+, Adults only 18+ and Unrated.
- Marks and Channels: Mark used is Line. Channel used is Length/Position. I chose this combination of mark and channel because I think, by using this combination, it is very easy to interpret the information given by the visualization.
- Type of variables used: The Y-axis has average rating which is numerical derived out of the numerical variables 'Rating Count' and 'Rating'. The X-axis has the categorical variable 'Content Rating'.
### Part B: Success of apps based on their content rating by calculating average maximum installs:
- Found the unique entries in the column 'Content Rating'.
- For each unique content rating, calculated the maximum installs = (summation(maximum installs))/(number of filtered rows). Here, summation implies the sum of the mentioned quantity over all the filtered rows.
- Made a bar graph of Average Maximum Installs vs Content Rating.
- Our hypothesis: The number 1 content rating based on average rating would be Everyone. Our hypothesis was corrrect.
- Inference: The order of the content rating, based on average rating, from topmost to bottommost is Everyone, Teen, Everyone 10+, Mature 17+, Adults only 18+ and Unrated.
- Marks and Channels: Mark used is Line. Channel used is Length/Position. I chose this combination of mark and channel because I think, by using this combination, it is very easy to interpret the information given by the visualization.
- Type of variables used: The Y-axis has average maxiumum installs which is numerical derived out of the numerical variable 'Maximum Installs'. The X-axis has the categorical variable 'Content Rating'.
- I have used average maximum installs instead of maximum installs to find the typical/most expected value of 'Maximum Installs'.

## Images in our LaTeX report and the corresponding .py file:
- Figure 17: visualization1.py
- Figure 18: visualization2.py
- Figure 19: visualization3.py
- Figure 20: visualization4.py

## How to run the code?
```bash
python processing.py
```

```bash
python visualization1.py
```

```bash
python visualization2.py
```

```bash
python visualization3.py
```

```bash
python visualization4.py
```

# Assignment 2
This section has 2 subsections: a) colour maps for sea surfaced data corresponding to the United States of America and b) tree maps, with user interaction, for the Google Playstores Apps dataset. The colour maps are made using Python libraries. The data for the used for creating the tree maps is preprocessed using Python and the webpage for user interaction is made using HTML, CSS and JavaScript.

## Contribution:
In the first subsection, I researched about the natural indices which can be derived from the columns in the sea surfaced dataset. After finding the appropriate indices, I standardized and scaled all the columns and applied the formulae to derive the natural indices. Then, I plotted the colour maps for these derived natural indices using Inferno colour maps. I have chosen the continuous scale amongst the continuous scale, discrete scale and logarithmic scale because on comparing the colour maps by using the three and analyzing the data I realized that the continuous scale is the best option. In the second subsection, I contributed in the preprocessing of the Google Playstores Apps dataset. Then, I wrote the code for making the webpage to enable user interaction. The user has to upload the .csv file, select the columns to be used and select the type of layout. Then when the user clicks on "Generate Treemap", the tree map is created and two colour sliders are provided to enable the user to highlight sections of the tree map which lie in a particular percentage range (using linear interpolation on the start and end of the range, captured by the positions of the colour sliders). Then, I selected combinations of the features of the dataset, which give a deeper insight into the dataset and drew inferences from the tree maps created.

## How to run the code:
### Make a Python virtual environment:
- Run this comand:
  ```bash
  python -m dvenv
  ```
- Then, run this command:
  ```bash
  dvenv\Scripts\activate
  ```
- Then, run this command:
  ```bash
  pip install -r requirements.txt
  ```
### Tree map visualizations:
- Navigate to the correct to the folder:
  ```bash
  cd '.\Assignment 2\Part 2 Treemap visualization\'
  ```
- Run the following command:
  ```bash
  python -m http.server
  ```
- Copy the HTTP URL output on the command the line and paste it on the browser. Replace '[::]' by 'localhost' in the URL and then press Enter. You will be able to see the folder structure on the webpage. Click on the playstore.html file. This will open the webpage. Now, you can use the webpage to generate tree maps by uploading .csv files.
### Colour map visulaizations:
- Navigate to the folder '.\Assignment 2\Part 1 sea surfaced data\'. Do not use the command line to navigate because a .ipynb needs to be run. Navigate manually on a file explorer to the folder mentioned.
- Now you will find the file DV.ipynb. Select the virtual environment you made (dvenv), as the kernel for this .ipynb file.
- Now run all the cells of this file.
