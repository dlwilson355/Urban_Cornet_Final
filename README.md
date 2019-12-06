# Urban Cornet Final
### Looking for relationships and patterns in what causes nations to be happy.

This project analyzes various factors such as educational quality, per capita wealth, and military spending to determing their relations to a nation's level of happiness.

Anaconda should come bundled with most of the libraries needed to run the source code.

Cartopy is required in order to be able to generate maps.  It can be installed on the conda-forge channel with...

**conda install -c conda-forge cartopy**

If you have an older version of anaconda that doesn't have seaborn by default, you will need to install it...

**conda install seaborn**

The source code for this project is contained in the following files...

- load_education.py: Contains code for loading the education data and performing plotting with it.

- military.py: Contains code for loading the military data and performing plotting with it.

- wealth_data.py: Contains code for loading the wealth data and performing plotting with it.

- happiness_relationships.py: Contains code for loading the happiness, and performing analyses and generating plots of how happiness relates to all our other data.

- create_maps_from_data.py: Contains code that generates world maps in which nations are color coded based on data.

- happiness_predictor.py: Contains code that uses all our datasets as input to train a model that predicts national happiness, and then plots the predicted vs actual happiness on a world map.


The following files provided functionality that is used by the other files in this project.

- stats.py: Contains code that can compute basic statistics on data sets when organized in a dictionary format.

- plots.py: Contains code that can make basic plots of our data sets when organized in a dictionary format.

- maps.py: Contains code that produces maps when passed data sets organized in a dictionary format.

- country_name_matching.py: Contains data structures and functions useful for matching different strings belonging to the same nation (i.e. United States with United States of America).


The data directory contains the datasets used for this project.  They are formatted as csv files.  This directory contains...

- WHR.csv: Contains happiness data along with other factors such as corruption, freedom, and life expectancy.

- education.csv: Contains information about the quality of education available in a country, including its primary school completation rate and literacy rate.

- gdp_data.csv: Contains data for the gross domestic product of each nation.

- milt_GDP_per.csv: Contains the data showing the military expenditure of each nation in terms of the percent of gross domestic product.

- milt_per_gov.csv: Contains the data showing the military expenditure of each nation in terms of the percent of government income spent on military.

- poverty_data.csv: Contains data showing the percent of inhabitants of the country living in poverty.
