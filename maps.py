"""
This file contains reusable functions for making maps of data.
The functions assume the data is passed as a dictionary where the keys are strings specifying the countries and the
values are the variable of interest for the corresponding nation.

Cartopy can be installed with "conda install -c conda-forge cartopy".

TODO: How to handle cases where strings are different.
"""


import copy
import random
import string

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap
import numpy as np

from country_name_info import MANUAL_MATCHES


def create_discrete_data_map(data_dict, title="World Map", order="auto"):
    """This function creates a world map with countries drawn on it."""

    # copy the dict so we don't change it
    data = copy.copy(data_dict)

    plt.style.use('Solarize_Light2')

    # create a projection
    ax = plt.axes(projection=ccrs.PlateCarree())

    # add coastlines
    ax.coastlines()

    # read country shape files
    shape_file_name = shpreader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries")
    reader = shpreader.Reader(shape_file_name)
    countries = reader.records()

    # color the countries by value
    if order == "auto":
        values = sorted(list(set(data.values())))
    else:
        values = order
    color_map = get_cmap("viridis", len(values))
    colors = [color_map(i / (len(values)-1)) for i in range(len(values))]
    for country in countries:
        match = get_matching_key(country.attributes["NAME_EN"], data)
        if match:
            ax.add_geometries(country.geometry,
                              ccrs.PlateCarree(),
                              facecolor=colors[values.index(data[match])],
                              edgecolor="#000000")

    # add a legend
    handles = []
    for color, value in zip(colors, values):
        handles.append(mpatches.Patch(color=color, label=value))
        plt.legend(handles=handles, loc='center left', fancybox=True)

    # add title
    plt.title(title)

    # display the plot
    plt.show()


def create_continuous_data_map(data_dict, title="World Map", color_bar_label="Values", log_x=False):
    """This function creates a world map with countries drawn on it."""

    # copy the dict so we don't change it
    data = copy.copy(data_dict)

    # take the log of the data if appropriate
    if log_x:
        for key in data.keys():
            if not data[key] == 0:
                data[key] = np.log(data[key])

    plt.style.use('Solarize_Light2')

    # create a projection
    fig = plt.figure(figsize=(8, 3))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # add coastlines
    ax.coastlines()

    # read country shape files
    shape_file_name = shpreader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries")
    reader = shpreader.Reader(shape_file_name)
    countries = reader.records()

    def get_0_to_1(value):
        """
        Returns a number 0 to 1 representing where the value lies relative to the rest of the plotted values.
        Used for determining what value to pass to the color map.
        """

        largest_value = max(data.values()) - min(data.values())
        this_value = value - min(data.values())
        zero_to_one = this_value / largest_value

        return zero_to_one

    # color the countries by value
    print("\n\nAttempting to match strings in data dictionary with strings from map data...")
    color_map = get_cmap("viridis", 1e3)
    for country in countries:
        match = get_matching_key(country.attributes["NAME_EN"], data)
        if match:
            ax.add_geometries(country.geometry,
                              ccrs.PlateCarree(),
                              facecolor=color_map(get_0_to_1(data[match])),
                              edgecolor="#000000")
        else:
            ax.add_geometries(country.geometry,
                              ccrs.PlateCarree(),
                              facecolor="#999999",
                              edgecolor="#000000")

    # add a colorbar
    cax = fig.add_axes([0.17, 0.20, 0.02, 0.5])
    color_bar = mpl.colorbar.ColorbarBase(ax=cax, cmap=color_map, boundaries=sorted(data.values()))
    color_bar.set_label(color_bar_label)

    # add title
    ax.set_title(title)

    # display the plot
    plt.show()


def get_matching_key(match_string, dict):
    """Returns a key whose string closely matches the passed string.  If no matches are found, False is returned."""

    string1 = match_string.lower().translate(str.maketrans('', '', string.punctuation))
    for key in dict.keys():
        string2 = key.lower().translate(str.maketrans('', '', string.punctuation))
        words_1 = string1.split()
        words_2 = string2.split()

        # if the two strings are similar match them
        if all(word in words_1 for word in words_2) or all(word in words_2 for word in words_1):
            print(f"Matched '{key}' with '{match_string}'.")
            return key

        # otherwise check for a manual match
        for matching_string in MANUAL_MATCHES:
            match = True
            for word in words_1 + words_2:
                if word not in matching_string.lower():
                    match = False
            if match:
                print(f"Matched '{key}' with '{match_string}'.")
                return key

    # otherwise no match is found
    print(f"No match found for {match_string}.")

    return False


def generate_discrete_sample_data(num_values=20):
    """Returns sample data with which to test mapping functions."""

    # read the available countries
    shape_file_name = shpreader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries")
    reader = shpreader.Reader(shape_file_name)
    countries = reader.records()

    # construct a dictionary from sample data
    data_dict = {}
    for country in countries:
        data_dict[country.attributes["NAME_EN"]] = f"Value {random.randint(1, num_values)}"

    # construct a sorted list of ordered values (optional)
    ordered_values = [f"Value {i+1}" for i in range(num_values)]

    return data_dict, ordered_values


def generate_continuous_sample_data():
    """Returns sample data with which to test mapping functions."""

    # read the available countries
    shape_file_name = shpreader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries")
    reader = shpreader.Reader(shape_file_name)
    countries = reader.records()

    # construct a dictionary from sample data
    data_dict = {}
    for country in countries:
        data_dict[country.attributes["NAME_EN"]] = random.uniform(0, 1e6)

    return data_dict


# running this file will produce sample maps
if __name__ == "__main__":

    # create a map of discrete national measurements
    data, order = generate_discrete_sample_data()
    create_discrete_data_map(data, order=order)

    # create a map of continuous national measurements
    data = generate_continuous_sample_data()
    create_continuous_data_map(data)
