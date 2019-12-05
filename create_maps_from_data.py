"""This file contains code for generating maps of the individual data sets."""


import numpy as np

from wealth_data import load_wealth_data, GDP_DATA_FILE_PATH, POVERTY_DATA_FILE_PATH
from happiness_relationships import read_happiness, read_military, TRANSLATE_DICT, get_info_dict
from load_education import load_data_education
import maps


def create_wealth_maps():
    """Creates the world maps for wealth."""

    all_data_dict = get_info_dict()
    print(all_data_dict)

    # create a maps of emotional affects, freedoms, life expectancy, ect
    variables_to_map = ["Ladder",
                        "Pos Affect",
                        "Neg Affect",
                        "Social Support",
                        "Freedom",
                        "Corruption",
                        "Generosity",
                        "GDP",
                        "Life Exp",
                        "GDP 2",
                        "Percent Military(of GDP)",
                        "Primary Completion",
                        "Literacy",
                        "Percent Military(of Gov)",
                        "Poverty"]

    map_names = ["Ladder Rating",
                 "Positive Affect Rating",
                 "Negative Affect Rating",
                 "Social Support Rating",
                 "Freedom Rating",
                 "Corruption Rating",
                 "Generosity Rating",
                 "GDP Ranking",
                 "Life Expectancy Rating",
                 "Gross Domestic Product (log scale)",
                 "Percent of GDP Spending on Military",
                 "Primary Completion Rate",
                 "Literacy Rate",
                 "Percent of Government Funds Spent on Military",
                 "Percent Poverty Rate"]

    use_log_scale = [False, False, False, False, False, False, False, False,
                     False, True, False, False, False, False, False]

    for variable, map_name, use_log in zip(variables_to_map, map_names, use_log_scale):
        data_dict = {key: value[variable] for (key, value) in all_data_dict.items() if not np.isnan(value[variable])}
        print(data_dict)
        maps.create_continuous_data_map(data_dict,
                                        f"World Map of {map_name}",
                                        f"{map_name}",
                                        use_log)


if __name__ == "__main__":
    create_wealth_maps()
