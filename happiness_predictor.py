"""
This file contains the machine learning algorithm intended to predict happiness in a country.

As input, the algorithm receives military spending data, GDP and poverty data, and education data.
As output, one of the algorithms predicts the positive affect of the country,
and the other algorithm predicts the negative affect of the country.
"""

import random

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler

from wealth_data import load_wealth_data, GDP_DATA_FILE_PATH, POVERTY_DATA_FILE_PATH
from happiness_relationships import read_happiness, read_military, happiness, keys
from load_education import load_data_education
from country_name_matching import get_matching_key
import maps
import stats
import plots
from country_name_matching import TRANSLATE_DICT


def load_all_data():
    """
    Loads a dictionary where each key is a string representing a country's name.
    Each value in the dictionary is a list of data measurements, where np.NAN is used in place of missing values.
    """

    # load GDP data
    gdp_data = load_wealth_data(GDP_DATA_FILE_PATH)

    # load poverty data
    poverty_data = load_wealth_data(POVERTY_DATA_FILE_PATH)

    # load happiness data
    happiness_data = read_happiness(keys + happiness)

    # load military government percent spending data
    military_data = read_military(r'data/milt_per_gov.csv', happiness_data, TRANSLATE_DICT, '(of Gov)', keys + happiness)
    military_govt_percent_data = {key: value['Percent Military(of Gov)']
                                  for (key, value) in military_data.items()
                                  if 'Percent Military(of Gov)' in value.keys()}

    # load government GDP percent military spending
    military_data = read_military(r'data/milt_per_gov.csv', happiness_data, TRANSLATE_DICT, '(of GDP)', keys + happiness)
    military_gdp_percent_data = {key: value['Percent Military(of GDP)']
                                 for (key, value) in military_data.items()
                                 if 'Percent Military(of GDP)' in value.keys()}

    # load primary school completion rate data
    variables_to_map = ["Primary completion rate, total (% of relevant age group)",
                        "Literacy rate, adult total (% of people ages 15 and above)"]
    education_data = load_data_education(r"data/education.csv", variables_to_map)
    primary_completion_rates = {key: float(value[2]) for (key, value) in education_data.items() if
                                not value[2] == "No Data"}

    # load literacy data
    literacy_rates = {key: float(value[5]) for (key, value) in education_data.items() if
                      not value[5] == "No Data"}

    # construct a dict of all data across all the data sets
    all_data_dict = {}
    for country in happiness_data.keys():

        # create the list of data for each country
        country_data = []

        # add the data from the happiness data set
        country_data.append(happiness_data[country]['Ladder'])
        country_data.append(happiness_data[country]['Pos Affect'])
        country_data.append(happiness_data[country]['Neg Affect'])
        country_data.append(happiness_data[country]['Social Support'])
        country_data.append(happiness_data[country]['Freedom'])
        country_data.append(happiness_data[country]['Corruption'])
        country_data.append(happiness_data[country]['Generosity'])
        country_data.append(happiness_data[country]['Life Exp'])

        # add the country data from the other data sets
        data_sets_to_merge = [gdp_data,
                              poverty_data,
                              military_gdp_percent_data,
                              military_govt_percent_data,
                              primary_completion_rates,
                              literacy_rates]

        # temporarily removing GDP data
        data_sets_to_merge = [poverty_data,
                              military_gdp_percent_data,
                              military_govt_percent_data,
                              primary_completion_rates,
                              literacy_rates]

        for data_set in data_sets_to_merge:
            match = get_matching_key(country, data_set, silent=True)
            if match:
                country_data.append(data_set[match])
            else:
                country_data.append(np.NAN)

        all_data_dict[country] = country_data

    return all_data_dict


def get_train_test_splits(data_dict, affect="Ladder", num_splits=10):
    """Returns a list of splits used for training and testing."""

    country_splits = split_countries(data_dict, num_splits)
    X_train_splits, X_test_splits, Y_train_splits, Y_test_splits = [], [], [], []

    for testing_countries in country_splits:

        X_train, X_test, Y_train, Y_test = [], [], [], []

        for country in data_dict.keys():

            # find the x and y data for the country
            x = data_dict[country][3:]
            if affect == "Ladder":
                y = data_dict[country][0]
            elif affect == "Positive":
                y = data_dict[country][1]
            else:
                y = data_dict[country][2]

            # append to the training or testing data
            if country in testing_countries:
                X_test.append(x)
                Y_test.append(y)
            else:
                X_train.append(x)
                Y_train.append(y)

        # add the splits
        X_train_splits.append(X_train)
        X_test_splits.append(X_test)
        Y_train_splits.append(Y_train)
        Y_test_splits.append(Y_test)

    return X_train_splits, X_test_splits, Y_train_splits, Y_test_splits, country_splits


def split_countries(data_dict, num_splits=10):
    """
    This function randomly splits the countries to specify which ones should serve as predictions for each iteration.
    """

    # randomly split the countries
    countries = list(data_dict.keys())
    random.shuffle(countries)
    country_splits = [[] for _ in range(num_splits)]
    while len(countries) > 0:
        for split in country_splits:
            if len(countries) > 0:
                split.append(countries[-1])
                countries = countries[:-1]

    return country_splits


def get_SGD_model():

    pipeline = Pipeline([
        ('imp', IterativeImputer()),
        ('tf', RobustScaler()),
        ('clf', SGDClassifier()),
    ])

    return pipeline


def evaluate_errors(predicted_dict, actual_dict, affect="Ladder"):
    """Prints out statistics and makes plots about the errors of the algorithm."""

    # compute the errors
    errors = {country: abs(predicted_dict[country] - actual_dict[country]) for country in predicted_dict.keys()}

    # print some statistics about the errors
    print(f"Mean error for {affect} was {stats.compute_mean(errors)}.")
    print(f"Standard deviation of error for {affect} was {stats.compute_stdev(errors)}.")
    print(f"Median error for {affect} was {stats.compute_median(errors)}.")

    # make some plots of the errors
    plots.create_histogram(errors,
                           "Absolute Error",
                           "Count",
                           f"Errors When Predicting {affect} Affect")

    plots.create_ECDF(errors,
                      "Absolute Error",
                      "Pr < X",
                      f"Error Probability When Predicting {affect} Affect")

    # make maps of errors
    maps.create_continuous_data_map(errors,
                                    f"World Map of Errors Predicting {affect} Affect",
                                    f"Error Predicting {affect} Affect",
                                    log_x=False,
                                    reverse_cmap=False)


def train_and_map():
    """
    This function trains a model to predict both positive and negative affect.
    It makes a map of the predicted vs actual affects.
    """

    affects_to_predict = ["Ladder", "Positive", "Negative"]

    for affect in affects_to_predict:

        # store the test data values and test prediction results from each set of testing data
        all_predictions = {}
        all_actual = {}

        # load all the data and split for training and testing
        data_dict = load_all_data()
        X_train_splits, X_test_splits, Y_train_splits, Y_test_splits, C_splits = \
            get_train_test_splits(data_dict, affect=affect, num_splits=10)

        for X_train, X_test, Y_train, Y_test, C_split in \
            zip(X_train_splits, X_test_splits, Y_train_splits, Y_test_splits, C_splits):

            # fit the model and make predictions
            model = get_SGD_model()
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)

            # add this set of predictions to the master dictionary
            all_predictions = dict(all_predictions,
                                   **{country: prediction for country, prediction in zip(C_split, Y_pred)})
            all_actual = dict(all_actual,
                              **{country: actual for country, actual in zip(C_split, Y_test)})

        # construct maps of predicted vs actual
        maps.create_continuous_data_map(all_predictions,
                                        f"World Map of Predicted {affect} Affect",
                                        f"Predicted {affect} Affect",
                                        log_x=False,
                                        reverse_cmap=True)

        maps.create_continuous_data_map(all_actual,
                                        f"World Map of Actual {affect} Affect",
                                        f"Actual {affect} Affect",
                                        log_x=False,
                                        reverse_cmap=True)

        # print statistics and make plots for predicted vs actual
        evaluate_errors(all_predictions, all_actual, affect=affect)


if __name__ == "__main__":
    train_and_map()
