"""
This file contains code that runs a clustering algorithm to find countries with similar characteristics.
The similar countries are then plotted on a world map.
"""

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation, SpectralClustering, SpectralCoclustering, MeanShift

from happiness_predictor import load_all_data
from maps import create_discrete_data_map


def get_transformer():
    """Returns the transformer used for the algorithm."""

    transformer = Pipeline([
        ('imp', IterativeImputer()),
        ('tf', RobustScaler()),
    ])

    return transformer


def perform_clustering(data, countries, algorithm="auto"):
    """
    Performs clustering on the passed data.
    Returns a dictionary indicating which cluster each nation was paired with.
    """

    if algorithm == "auto":
        algorithm = KMeans(n_clusters=5)

    transformer = get_transformer()
    cluster_data = transformer.fit_transform(data)
    clusters = algorithm.fit_predict(cluster_data)
    data_dict = {country: f"Cluster {cluster + 1}" for (country, cluster) in zip(countries, clusters)}

    return data_dict


def load_country_data():
    """
    Loads all of the country data.
    Returns a tuple.
    The first item in the tuple is a list of lists of data.
    The second item in the tuple is a list of corresponding countries.
    """

    all_data = load_all_data()

    countries = list(all_data.keys())
    values = [all_data[country] for country in countries]

    return values, countries


if __name__ == "__main__":
    data, countries = load_country_data()

    algorithms_to_test = [DBSCAN(eps=.5),
                          AffinityPropagation(damping=0.7),
                          SpectralClustering(n_clusters=5),
                          MeanShift(bandwidth=2.5),
                          KMeans(n_clusters=5)]

    for algorithm in algorithms_to_test:
        data_dict = perform_clustering(data, countries, algorithm)
        create_discrete_data_map(data_dict, "World Map with Clusters Determined by Similar Characteristics")
