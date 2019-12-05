"""This file contains code for generating maps of the individual data sets."""


from wealth_data import load_wealth_data, GDP_DATA_FILE_PATH, POVERTY_DATA_FILE_PATH
from happiness_relationships import read_happiness, read_military, happiness, keys
from load_education import load_data_education
from country_name_matching import TRANSLATE_DICT
import maps


def create_maps():
    """Creates the world maps from each data set."""

    # create a GDP map
    gdp_data = load_wealth_data(GDP_DATA_FILE_PATH)
    maps.create_continuous_data_map(gdp_data,
                                    "World Map of National GDP",
                                    "Log of National GDP",
                                    log_x=True,
                                    reverse_cmap=False)

    # create a poverty map
    poverty_data = load_wealth_data(POVERTY_DATA_FILE_PATH)
    maps.create_continuous_data_map(poverty_data,
                                    "World Map of National Poverty",
                                    "Log of National Poverty Percentage",
                                    log_x=True,
                                    reverse_cmap=False)

    # create a maps of emotional affects, freedoms, life expectancy, ect
    variables_to_map = ["Ladder", "Pos Affect", "Neg Affect", "Social Support",
                        "Freedom", "Corruption", "Generosity", "Life Exp"]
    map_names = ["Ladder Rating", "Positive Affect Rating", "Negative Affect Rating", "Social Support Rating",
                 "Freedom Rating", "Corruption Rating", "Generosity Rating", "Life Expectancy Rating"]
    happiness_data = read_happiness(keys + happiness)
    for variable, map_name in zip(variables_to_map, map_names):
        maps.create_continuous_data_map({key: value[variable] for (key, value) in happiness_data.items()},
                                        f"World Map of {map_name}",
                                        f"{map_name}",
                                        log_x=False,
                                        reverse_cmap=True)

    # create maps of percent government spending
    military_data = read_military(r'data/milt_per_gov.csv', happiness_data, TRANSLATE_DICT, '(of Gov)', keys + happiness)
    military_govt_percent_data = {key: value['Percent Military(of Gov)']
                                  for (key, value) in military_data.items()
                                  if 'Percent Military(of Gov)' in value.keys()}
    maps.create_continuous_data_map(military_govt_percent_data,
                                    "World Map of Government Percent Military Spending",
                                    "Percent Spending",
                                    log_x=False,
                                    reverse_cmap=False)

    # create maps of government GDP spending
    military_data = read_military(r'data/milt_per_gov.csv', happiness_data, TRANSLATE_DICT, '(of GDP)', keys + happiness)
    military_gdp_percent_data = {key: value['Percent Military(of GDP)']
                                 for (key, value) in military_data.items()
                                 if 'Percent Military(of GDP)' in value.keys()}
    maps.create_continuous_data_map(military_gdp_percent_data,
                                    "World Map of Military GDP Percentage Spending",
                                    "Percent Spending",
                                    log_x=False,
                                    reverse_cmap=False)

    # create map of primary completion rate
    variables_to_map = ["Primary completion rate, total (% of relevant age group)",
                        "Literacy rate, adult total (% of people ages 15 and above)"]
    education_data = load_data_education(r"data/education.csv", variables_to_map)
    primary_completion_rates = {key: float(value[2]) for (key, value) in education_data.items() if
                                not value[2] == "No Data"}
    maps.create_continuous_data_map(primary_completion_rates,
                                    "World Map of Primary School Completion Rate",
                                    "Percent Completion",
                                    log_x=False,
                                    reverse_cmap=False)

    # create map of literacy rate
    literacy_rates = {key: float(value[5]) for (key, value) in education_data.items() if
                      not value[5] == "No Data"}
    maps.create_continuous_data_map(literacy_rates,
                                    "World Map of Literacy Rate",
                                    "Percent Literacy",
                                    log_x=False,
                                    reverse_cmap=False)


if __name__ == "__main__":
    create_maps()
