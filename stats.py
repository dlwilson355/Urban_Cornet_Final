"""
This file contains reusable functions for analyzing statistics of data.
The functions assume the data is passed as a dictionary where the keys are strings specifying the countries and the
values are the variable of interest for the corresponding nation.
"""


import statistics as st


def compute_mean(data_dict):
    """Returns the mean of the values in the data dict."""

    return st.mean(data_dict.values())


def compute_stdev(data_dict):
    """Returns the standard deviation data dict."""

    return st.stdev(data_dict.values())


def compute_median(data_dict):
    """Returns the median of the values in the data dict."""

    return st.median(data_dict.values())


# def compute_quantiles(data_dict):
#     """Returns the quantiles of the values in the data dict."""
#
#     return st.quantiles(data_dict.values())
