"""
This file contains code for loading national GDP data from the file in the file path specified at the beginning.
The GDP data can be downloaded at: https://data.worldbank.org/indicator/NY.GDP.MKTP.CD.
The poverty data can be downloaded at: https://data.worldbank.org/indicator/SI.POV.DDAY?locations=1W&start=1981&end=2015&view=chart.
"""


import random

import pandas as pd
import numpy as np

import stats
import plots
import maps
from country_name_info import COUNTRIES_TO_IGNORE


# the file path of the data to be loaded
GDP_DATA_FILE_PATH = r"data/gdp_data.csv"
POVERTY_DATA_FILE_PATH = r"data/poverty_data.csv"


def load_wealth_data(file_path):
    """Returns a dict where each key is a country and each value is the most recent measurement from the data file."""

    # load the data set and initialize an empty dict
    df = pd.read_csv(file_path)
    data = {}

    # iterate through each country
    for row in df.iterrows():

        # find the country's name
        country_name = row[1]['Country Name']
        if country_name not in COUNTRIES_TO_IGNORE:

            # find the most recent measurement
            year = 2020
            value = np.NaN
            while np.isnan(value) and year > 1960:
                year -= 1
                value = row[1][str(year)]

            # add the country to the dictionary
            if not np.isnan(value):
                data[country_name] = value

    return data


if __name__ == "__main__":

    # load the GDP data
    gdp_data = load_wealth_data(GDP_DATA_FILE_PATH)
    poverty_data = load_wealth_data(POVERTY_DATA_FILE_PATH)
    nations_of_interest = ["United States", "Finland", "Central African Republic", "Paraguay", "Turkey"]
    print(gdp_data)

    # compute the means
    print(f"Mean GDP is {stats.compute_mean(gdp_data)}.")
    print(f"Mean poverty percentage is {stats.compute_mean(poverty_data)}.")

    # compute the standard deviations
    print(f"The standard deviation for GDP is {stats.compute_stdev(gdp_data)}.")
    print(f"The standard deviation for poverty percentage is {stats.compute_stdev(poverty_data)}.")

    # compute the medians
    print(f"Median GDP is {stats.compute_median(gdp_data)}.")
    print(f"Median poverty percentage is {stats.compute_median(poverty_data)}.")

    # compute the quantiles
    # print(f"Mean GDP is {stats.compute_quantiles(gdp_data)}.")
    # print(f"Mean poverty percentage is {stats.compute_quantiles(poverty_data)}.")

    # create ECDF graphs
    plots.create_ECDF(gdp_data,
                      "Log of National GDP",
                      "Probability < X",
                      "Empirical Cumulative Distribution of National GDP",
                      True,
                      nations_of_interest)

    plots.create_ECDF(poverty_data,
                      "Log of National Poverty",
                      "Probability < X",
                      "Empirical Cumulative Distribution of National Poverty",
                      True,
                      nations_of_interest)

    # create histograms
    plots.create_histogram(gdp_data,
                           "Log of National GDP",
                           "Count",
                           "Histogram of National GDP",
                           True)

    plots.create_histogram(poverty_data,
                           "Log of National Poverty",
                           "Count",
                           "Histogram of National Poverty",
                           True)

    # create a scatter plot
    plots.create_scatter_plot(gdp_data,
                              poverty_data,
                              "Log of National GDP",
                              "Log of National Poverty",
                              "Scatter Plot of National GDP vs Poverty",
                              True,
                              True,
                              nations_of_interest)

    # create maps
    maps.create_continuous_data_map(gdp_data,
                                    "World Map of National GDP",
                                    "Log of National GDP",
                                    True)

    maps.create_continuous_data_map(poverty_data,
                                    "World Map of National Poverty",
                                    "Log of National Poverty Percentage",
                                    True)
