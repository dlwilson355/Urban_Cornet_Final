"""
This file contains reusable functions for making maps of data.
The functions assume the data is passed as a dictionary where the keys are strings specifying the countries and the
values are the variable of interest for the corresponding nation.

TODO: How to handle cases where strings are different.
"""


import copy
import random

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap
import numpy as np


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

    # color the countries by value
    print("\n\nAttempting to match strings in data dictionary with strings from map data...")
    color_map = get_cmap("viridis", 1e3)
    for country in countries:
        match = get_matching_key(country.attributes["NAME_EN"], data)
        if match:
            ax.add_geometries(country.geometry,
                              ccrs.PlateCarree(),
                              facecolor=color_map(data[match] / max(data.values())),
                              edgecolor="#000000")
        else:
            ax.add_geometries(country.geometry,
                              ccrs.PlateCarree(),
                              facecolor="#999999",
                              edgecolor="#000000")

    # add a colorbar
    cax = fig.add_axes([0.15, 0.17, 0.02, 0.5])
    color_bar = mpl.colorbar.ColorbarBase(ax=cax, cmap=color_map, boundaries=sorted(data.values()))
    color_bar.set_label(color_bar_label)

    # add title
    ax.set_title(title)

    # display the plot
    plt.show()


def get_matching_key(string, dict):
    """Returns a key whose string closely matches the passed string.  If no matches are found, False is returned."""

    string1 = string.lower()
    for key in dict.keys():
        string2 = key.lower()
        if string1 in string2 or string2 in string1:
            print(f"Matched '{key}' with '{string}'.")
            return key

    print(f"No match found for {string}.")

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
