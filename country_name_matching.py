"""
This file contains a list of countries listed in some of our data sets that aren't actually countries or aren't
suitable for analysis.  The list is used to determine which countries should be ignored when loading the data sets.
"""


import string


COUNTRIES_TO_IGNORE = ["World",
                       "Upper middle income",
                       "Sub-Saharan Africa (IDA & IBRD countries)",
                       "Middle East & North Africa (IDA & IBRD countries)",
                       "Latin America & the Caribbean (IDA & IBRD countries)",
                       "Europe & Central Asia (IDA & IBRD countries)",
                       "East Asia & Pacific (IDA & IBRD countries)",
                       "Small states",
                       "Sub-Saharan Africa (excluding high income)",
                       "South Asia",
                       "Post-demographic dividend",
                       "Pacific island small states",
                       "Pre-demographic dividend",
                       "Other small states",
                       "OECD members",
                       "North America",
                       "Middle income",
                       "Middle East & North Africa (excluding high income)",
                       "Middle East & North Africa",
                       "Late-demographic dividend",
                       "Low & middle income",
                       "Lower middle income",
                       "Low income",
                       "Least developed countries: UN classification",
                       "Latin America & Caribbean",
                       "Latin America & Caribbean (excluding high income)",
                       "IDA only",
                       "IDA blend",
                       "IDA total",
                       "IDA & IBRD total",
                       "IBRD only",
                       "Heavily indebted poor countries (HIPC)",
                       "High income",
                       "Fragile and conflict affected situations",
                       "European Union",
                       "Euro area",
                       "Europe & Central Asia",
                       "Europe & Central Asia (excluding high income)",
                       "East Asia & Pacific",
                       "Early-demographic dividend",
                       "East Asia & Pacific (excluding high income)",
                       "Caribbean small states",
                       "Central Europe and the Baltics",
                       "Arab World",
                       "Tuvalu",
                       "Nauru",
                       "Kiribati",
                       "Marshall Islands",
                       "Palau",
                       "Micronesia, Fed. Sts.",
                       "Sao Tome and Principe",
                       "Tonga",
                       "Sub-Saharan Africa"]


MANUAL_MATCHES = ["Congo Dem Rep Democratic Republic of the Congo (Brazzaville) (Kinshasa)",
                  "Russia Russian Federation",
                  "South Korea Rep",
                  "Cote dIvoire Ivory Coast",
                  "Slovakia Solvak Republic",
                  "Syria Syrian Arab Republic",
                  "Kyrgyz Republic Kyrgyzstan",
                  "Lao PDR Laos",
                  "Somalia Somaliland",
                  "Taiwan Republic of China",
                  "North Macedonia Republic of",
                  "Swaziland Kingdom of Eswatini"]


TRANSLATE_DICT = {'Lao PDR':'Laos','Venezuela, RB':'Venezuela',
                  'Iran, Islamic Rep.':'Iran','Gambia, The': 'Gambia',
                  'Syrian Arab Republic':'Syria','North Macedonia':'Macedonia',
                  'Bosnia and Herzegovina': 'Bosnia and Herzegovina ',
                  'Russian Federation': 'Russia','Slovak Republic': 'Slovakia',
                  'Central African Rep.': 'Central African Republic',
                  "Cote d'Ivoire": 'Ivory Coast',
                  'Côte d’Ivoire': 'Ivory Coast', 'eSwatini': 'Swaziland',
                  'Eswatini':'Swaziland','Congo, Dem. Rep.' : 'Congo (Kinshasa)',
                  'Congo, Rep.' : 'Congo (Brazzaville)',
                  'Dominican Rep.': 'Dominican Republic',
                  'Korea, Rep.' : 'South Korea',
                  'Trinidad & Tobago': 'Trinidad and Tobago',
                  'Kyrgyz Republic': 'Kyrgyzstan',
                  'Hong Kong SAR, China': 'Hong Kong', 'Yemen, Rep.' : 'Yemen',
                  'USA': 'United States', 'United States of America' : 'United States',
                  'Viet Nam': 'Vietnam', 'UK': 'United Kingdom',
                  'UAE': 'United Arab Emirates', 'Egypt, Arab Rep.': 'Egypt'}


def get_matching_key(match_string, data_dict, silent=True):
    """
    This function is used to matching strings referring to the same nation.
    It searches through the data_dict to find a country that matches the match_string.
    It returns the key in data_dict that matches, and returns False if no match is found.
    """

    string1 = match_string.lower().translate(str.maketrans('', '', string.punctuation))
    for key in data_dict.keys():
        string2 = key.lower().translate(str.maketrans('', '', string.punctuation))
        words_1 = string1.split()
        words_2 = string2.split()

        # if the two strings are similar match them
        if all(word in words_1 for word in words_2) or all(word in words_2 for word in words_1):
            if not silent:
                print(f"Matched '{key}' with '{match_string}'.")
            return key

        # otherwise check for a manual match
        for matching_string in MANUAL_MATCHES:
            match = True
            for word in words_1 + words_2:
                if word not in matching_string.lower():
                    match = False
            if match:
                if not silent:
                    print(f"Matched '{key}' with '{match_string}'.")
                return key

    # otherwise no match is found
    if not silent:
        print(f"No match found for {match_string}.")

    return False



