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
- Navigate to the folder '.\Assignment 2\Part 1 sea surfaced data\'. Do not use the command line to navigate because a .ipynb needs to be run. Navigate manually using a file explorer to the folder mentioned.
- Now you will find the file DV.ipynb. Select the virtual environment you made (dvenv), as the kernel for this .ipynb file.
- Now run all the cells of this file.
- To view the GIFs corresponding to the natural indices, navigate to the folder: "Assignment 2\Part 1 sea surfaced data". Please use a file explorer to navigate to the folder. There are folders named with each natural index. In each folder, there is a GIF.