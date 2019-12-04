"""
This file contains the machine learning algorithm intended to predict happiness in a country.

As input, the algorithm receives military spending data, GDP and poverty data, and education data.
As output, one of the algorithms predicts the positive affect of the country,
and the other algorithm predicts the negative affect of the country.
"""

import random

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import normalize, RobustScaler, StandardScaler

from wealth_data import load_wealth_data, GDP_DATA_FILE_PATH, POVERTY_DATA_FILE_PATH
from WHR_EDA import read_happiness, read_military, TRANSLATE_DICT
from load_education import load_data_education
from country_name_matching import get_matching_key
from maps import create_continuous_data_map


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
    happiness_data = read_happiness()

    # load military government percent spending data
    military_data = read_military(r'data/milt_per_gov.csv', read_happiness(), TRANSLATE_DICT, '(of Gov)')
    military_govt_percent_data = {key: value['Percent Military(of Gov)']
                                  for (key, value) in military_data.items()
                                  if 'Percent Military(of Gov)' in value.keys()}

    # load government GDP percent military spending
    military_data = read_military(r'data/milt_per_gov.csv', read_happiness(), TRANSLATE_DICT, '(of GDP)')
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

        for data_set in data_sets_to_merge:
            match = get_matching_key(country, data_set, silent=True)
            if match:
                country_data.append(data_set[match])
            else:
                country_data.append(np.NAN)

        all_data_dict[country] = country_data

    return all_data_dict


def get_train_test_split(data_dict, affect="Positive", testing_proportion=0.25):
    """
    Takes a data_dict, which type of Y data is desired, and a testing proportion as input.
    Returns numpy arrays of X and Y data along with a "lookup list" indicated which values correspond to which nations.
    """

    # determine how many training and testing data point to make
    num_testing = int(testing_proportion * len(data_dict))
    num_training = len(data_dict) - num_testing

    # randomly order the countries
    countries = list(data_dict.keys())
    random.shuffle(countries)

    # create the training and testing data
    X_train, X_test, Y_train, Y_test, countries_train, countries_test = [], [], [], [], [], []
    for country in countries:

        # find the x and y data for the country
        x = data_dict[country][2:]
        if affect == "Positive":
            y = data_dict[country][0]
        else:
            y = data_dict[country][1]

        # append to the training or testing data
        if len(X_train) < num_training:
            X_train.append(x)
            Y_train.append(y)
            countries_train.append(country)
        else:
            X_test.append(x)
            Y_test.append(y)
            countries_test.append(country)

    return np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test), countries_train, countries_test


def get_SGD_grid_search_model():
    """Returns the model used to prediction national happiness."""

    pipeline = Pipeline([
        ('clf', SGDClassifier()),
    ])

    params = {
        'clf__loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),
        'clf__alpha': (1e-2, 1e-3),
        'clf__penalty': ('none', 'l1', 'l2', 'elasticnet'),
        'clf__early_stopping': (True, False),
    }

    model = GridSearchCV(pipeline, params, cv=10, n_jobs=-1)

    return model


def get_SGD_model():

    pipeline = Pipeline([
        ('imp', IterativeImputer()),
        ('tf', RobustScaler()),
        ('clf', SGDClassifier()),
    ])

    return pipeline


def train_and_map():
    """
    This function trains a model to predict both positive and negative affect.
    It makes a map of the predicted vs actual affects.
    """

    affects_to_predict = ["Positive", "Negative"]

    for affect in affects_to_predict:
        # load all the data and split for training and testing
        data_dict = load_all_data()
        X_train, X_test, Y_train, Y_test, countries_train, countries_test = \
            get_train_test_split(data_dict, affect=affect, testing_proportion=0.5)

        # fit the model and make predictions
        model = get_SGD_model()
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        # make map of predicted affect
        predictions = {country: prediction for country, prediction in zip(countries_test, Y_pred)}
        create_continuous_data_map(predictions,
                                   f"World Map of Predicted {affect} Affect",
                                   f"Predicted {affect} Affect",
                                   False)

        # make map of actual affect
        actual = {country: actual for country, actual in zip(countries_test, Y_test)}
        create_continuous_data_map(actual,
                                   f"World Map of Actual {affect} Affect",
                                   f"Actual {affect} Affect",
                                   False)


if __name__ == "__main__":
    train_and_map()
