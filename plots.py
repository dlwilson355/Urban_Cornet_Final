"""
This file contains reusable functions for making plots of data.
The functions assume the data is passed as a dictionary where the keys are strings specifying the countries and the
values are the variable of interest for the corresponding nation.
"""

import copy
import random

import matplotlib.pyplot as plt
import numpy as np


def create_ECDF(data_dict,
                x_label="X",
                y_label="Y",
                title="ECDF Plot",
                log_x=False,
                marks=[]):

    """
    Plots the empirical cumulative distribution of the values in the data dict.

    The required argument is a data dict (as described at the beginning of the file).
    Optional arguments are axis labels, plot title, a boolean determining whether the log of x values are plotted, and a
    list of nations to mark on the graph.
    """

    # copy the dict so we don't change it
    data = copy.copy(data_dict)

    # take the log of the data if appropriate
    if log_x:
        for key in data.keys():
            if not data[key] == 0:
                data[key] = np.log(data[key])

    # create the list of X and Y data to plot
    X = sorted(data.values())
    Y = np.arange(len(X)) / len(X)

    # plot the ECDF
    plt.style.use('Solarize_Light2')
    plt.plot(X, Y, '.', markersize=20)

    # plot vertical lines and a legend marking nations of interest
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    if len(marks) > 0:
        for color, nation in zip(colors, marks):
            plt.axvline(data[nation], color=color, label=nation)
        plt.legend()

    # add titles
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # display the plot
    plt.show()


def create_histogram(data_dict,
                     x_label="X",
                     y_label="Count",
                     title="Histogram",
                     log_x=False,
                     bins="auto"):

    """
    Plots a histogram of the values in the data dict.

    The required argument is a data dict (as described at the beginning of the file).
    Optional arguments are axis labels, plot title, a boolean determining whether the log of x values are plotted,
    and bin size.
    """

    # copy the dict so we don't change it
    data = copy.copy(data_dict)

    # take the log of the data if appropriate
    if log_x:
        for key in data.keys():
            if not data[key] == 0:
                data[key] = np.log(data[key])

    # create the list of X data
    X = list(data.values())

    # plot the histogram
    plt.style.use('Solarize_Light2')
    plt.hist(X, bins=bins)

    # add titles
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # display the plot
    plt.show()


def create_scatter_plot(x_data_dict,
                        y_data_dict,
                        x_label="X",
                        y_label="Y",
                        title="Scatter Plot",
                        log_x=False,
                        log_y=False,
                        marks=[]):

    """
    Plots a scatter plot of the values in the data dicts.

    The required argument is a data dict (as described at the beginning of the file).
    Optional arguments are axis labels, plot title, a boolean determining whether the log of x values are plotted, and a
    list of nations to mark on the graph.
    """

    # copy the dict so we don't change it
    x_data = copy.copy(x_data_dict)
    y_data = copy.copy(y_data_dict)

    # take the log of the data as appropriate
    if log_x:
        for key in x_data.keys():
            if not x_data[key] == 0:
                x_data[key] = np.log(x_data[key])

    if log_y:
        for key in y_data.keys():
            if not y_data[key] == 0:
                y_data[key] = np.log(y_data[key])

    # create a list of countries common to both data sets
    countries = set([key for key in x_data.keys() if key in y_data] + [key for key in y_data.keys() if key in x_data])

    # create lists of X, Y, marked_X, and marked_Y data
    X = [x_data[key] for key in countries if key not in marks]
    Y = [y_data[key] for key in countries if key not in marks]
    marked_X = [x_data[key] for key in marks]
    marked_Y = [y_data[key] for key in marks]

    # plot the scatter plot
    plt.style.use('Solarize_Light2')
    fig, ax = plt.subplots()
    plt.scatter(X, Y)

    # plot and annotate the points of interest
    if len(marks) > 0:
        plt.scatter(marked_X, marked_Y, marker="D")
        for mark in marks:
            mark_X = x_data[mark]
            mark_Y = y_data[mark]
            mark_Y += max(max(Y), max(marked_Y))/60
            ax.annotate(mark,
                        (mark_X, mark_Y),
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # add titles
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # display the plot
    plt.show()


def generate_sample_data(num_samples=100, num_marks=5):
    """Returns sample data with which to test plotting functions."""

    X = np.random.normal(100, 25, (num_samples,))
    Y = [f"Nation {i+1}" for i in range(num_samples)]
    data_dict = dict(zip(Y, X))
    nations_to_mark = [random.choice(list(data_dict.keys())) for i in range(num_marks)]

    return data_dict, nations_to_mark


# running this file will produce samples of each plot
if __name__ == "__main__":

    # generate sample data
    sample_data_dict_1, sample_marks_1 = generate_sample_data()
    sample_data_dict_2, _ = generate_sample_data()

    # create an ECDF
    create_ECDF(sample_data_dict_1, marks=sample_marks_1)

    # create a histogram
    create_histogram(sample_data_dict_1)

    # create a scatter plot
    create_scatter_plot(sample_data_dict_1, sample_data_dict_2, marks=sample_marks_1)
