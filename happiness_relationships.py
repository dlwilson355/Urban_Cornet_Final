#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 20:25:27 2019
@author: Urban Cornet
"""
##############################################################################
'''imports '''
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import statistics
from wealth_data import load_wealth_data
from load_education import load_data_education
from country_name_matching import TRANSLATE_DICT
import seaborn as sn
import pandas as pd
##############################################################################
'''functions'''
##############################################################################
'''read in data'''

happiness = ['Ladder', 'Pos Affect', 'Neg Affect']
keys = [
    'Social Support', 'Freedom', 'Corruption', 'Generosity', 'GDP', 'GDP 2',
    'Life Exp', 'Percent Military(of GDP)', 'Primary Completion', 'Literacy',
    'Percent Military(of Gov)', 'Poverty'
]
ranks = [
        'Pos Affect', 'Neg Affect', 'Ladder', 'Social Support', 'Freedom',
        'Corruption', 'Generosity', 'GDP', 'Life Exp'
    ]

#TODO: combine these functions
def read_happiness(keys):
    '''read in happiness dataset. Return Dictionary of tidy data'''
    i = 0
    info_dict = {}
    f = open('data/WHR.csv')
    for line in f:
        i = i + 1
        if i > 3:
            line = line.strip().split(',')
            try:
                info_dict[line[0]] = {
                    'Ladder': int(line[1]),
                    'Pos Affect': int(line[3]),
                    'Neg Affect': int(line[4]),
                    'Social Support': int(line[5]),
                    'Freedom': int(line[6]),
                    'Corruption': int(line[7]),
                    'Generosity': int(line[8]),
                    'GDP': int(line[9]),
                    'Life Exp': int(line[10])
                }
            except:
                #no data. Make zero. Do not include in statistics and charts (piecewise)
                for i in range(len(line)):
                    if len(line[i]) < 1:
                        line[i] = np.NAN
                line = [line[0]] + [int(x) if not np.isnan(float(x)) else np.NAN for x in line[1:]]
                info_dict[line[0]] = {
                    'Ladder': line[1],
                    'Pos Affect': line[3],
                    'Neg Affect': line[4],
                    'Social Support': line[5],
                    'Freedom': line[6],
                    'Corruption': line[7],
                    'Generosity': line[8],
                    'GDP': line[9],
                    'Life Exp': line[10]
                }
    info_dict = reconcile(info_dict, keys)
    return info_dict


def read_education(trans, info, keys):
    '''read in education dataset, using given data dictionary. Return 
    dictionary of tidy data'''
    variables = [
        "Primary completion rate, total (% of relevant age group)",
        "Literacy rate, adult total (% of people ages 15 and above)"
    ]
    education_data = load_data_education('data/education.csv', variables)

    for item in education_data.keys():
        if item in trans.keys():
            jtem = trans[item]
        else:
            jtem = item
        try:
            info[jtem]['Primary Completion'] = float(education_data[item][2])
            info[jtem]['Literacy'] = float(education_data[item][5])
        except:
            pass
    info = reconcile(info, keys)
    return info


def read_data(key, filename, trans, info, keys):
    '''read in GDP and Poverty datasets, using given data dictionary. Return
    dictionary of tidy data.'''
    data = load_wealth_data(filename)

    for item in data.keys():
        if item in trans.keys():
            jtem = trans[item]
        else:
            jtem = item
        try:
            info[jtem][key] = data[item]
        except:
            pass
    info = reconcile(info, keys)
    return info


def read_military(filename, info_dict, translate, percent_of, keys):
    '''read in military dataset, using given dictionary. Return dictionary of 
    tidy data'''
    f = open(filename)
    for line in f:
        #no longer a country or heading line
        if 'xxx' in line or '%' not in line:
            continue
        line = line.strip()
        line = line.split(',')

        #missing data
        if '. .' in line[-1]:
            line[-1] = '0'
        country = line[0].strip('"')
        percent = float(line[-1].strip('%'))

        #need translation dictionary
        if country in list(translate.keys()):
            country = translate[country]

        if country in info_dict.keys():
            info_dict[country]['Percent Military' + percent_of] = percent
        else:
            pass

    info_dict = reconcile(info_dict, keys)
    return info_dict


def reconcile(info_dict, keys):
    '''input zero for missing data. This function acts on the assumption that
    colombia has all the keys'''
    for item in info_dict.keys():
        for jtem in keys:
            if jtem not in info_dict[item].keys():
                info_dict[item][jtem] = np.NAN
    return info_dict


##############################################################################
'''Data Analysis'''


def get_var(info_dict, key, marks):
    '''Get specific variable from tidy data dictionary'''
    temp = []
    mark_vals = {}
    for country, values in info_dict.items():
        temp.append(values[key])
        if country in marks:
            mark_vals[country] = values[key]
    return temp, mark_vals


def pairwise_delete(x, y):
    '''Pairwise delete missing information for relationship statistics'''
    x_copy = x[:]
    y_copy = y[:]
    for i in range(len(x)):
        if np.isnan(x[i]) or np.isnan(y[i]):
            x_copy.remove(x[i])
            y_copy.remove(y[i])
    return x_copy, y_copy


def plot_relationships(info_dict, x_key, y_key, ranks):
    '''Plot scatterplot between two variables. Pairwise delete missing data.
    Plot linear regression line. Adjust axes for ranked data. 
    Mark countries of interest (determined through EDA)'''
    marks = [
        'United States', 'Finland', 'Central African Republic', 'Paraguay',
        'Turkey', 'Taiwan'
    ]
    x, mark_x = get_var(info_dict, x_key, marks)
    y, mark_y = get_var(info_dict, y_key, marks)
    x, y = pairwise_delete(x, y)
    
    #linear regression
    m, b, r, p, sd = scipy.stats.linregress(x, y)
    line = [m * x1 + b for x1 in x]
    fig, ax = plt.subplots()
    if bool(x_key not in ranks or x_key == 'Corruption') ^ bool(y_key not in ranks or y_key =='Corruption'):
        textstr = 'r-value: ' + str(round(-1*r, 4)) + '\n p-value: ' + str(
            round(p, 4))
    else:
        textstr = 'r-value: ' + str(round(r, 4)) + '\n p-value: ' + str(
            round(p, 4))     
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    switches = ['Percent Military(of Gov)', 'Corruption','Percent Military(of GDP)',
                'Poverty']
    if x_key in switches:
        ax.text(
            0.6,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=props) 
    else:
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=props)

    #adjust axes
    if x_key in ranks and x_key != 'Corruption':
        ax.set_xlim(max(x) + 10, min(x) - 10)
        plt.xlabel(x_key + ' Ranking')
    else:
        plt.xlabel(x_key)
    if y_key in ranks and y_key != 'Corruption':
        ax.set_ylim(max(y) + 10, min(y) - 10)
        plt.ylabel(y_key + ' Ranking')
    else:
        plt.ylabel(y_key)
    plt.scatter(x, y)

    #mark countries
    plt.scatter(list(mark_x.values()), list(mark_y.values()))
    for value in mark_x.keys():
        if not np.isnan(mark_x[value]) and not np.isnan(mark_y[value]):
            props = dict(boxstyle='round', facecolor='white', alpha=0.50)
            ax.annotate(
                value, (mark_x[value], mark_y[value]), bbox=props)

    #plot
    plt.title(x_key + ' versus ' + y_key)
    plt.plot(x, line, 'r')
#    plt.show()
    plt.savefig('data/relationship_' + x_key + y_key,  bbox_inches='tight')


#TODO: add ranks to reference info
def find_trends(info_dict, keys, ranks):
    '''break into categories (extremely low, low, below average, above average, high, extremely high)
    and return dictionary of each country with its categorized variables'''

    ranks = [
        'Pos Affect', 'Neg Affect', 'Ladder', 'Social Support', 'Freedom',
        'Corruption', 'Generosity', 'GDP', 'Life Exp'
    ]
    keys = [
        'Pos Affect', 'Neg Affect', 'Ladder', 'Social Support', 'Freedom',
        'Corruption', 'Generosity', 'GDP', 'Life Exp',
        'Percent Military(of GDP)', 'Percent Military(of Gov)', 'Poverty',
        'Primary Completion', 'Literacy', 'GDP 2'
    ]
    data = {}
    categories = {}
    for val in keys:
        info, _ = get_var(info_dict, val, [])
        data[val] = info
    for item in data.keys():
        if item in ranks:
            categories[item] = [(len(data[item])) / 6, (len(data[item])) / 3,
                                len(data[item]) / 2, (2 * len(data[item])) / 3,
                                (5 * len(data[item])) / 6]
        else:
            average = statistics.mean(data[item])
            sd = statistics.stdev(data[item])
            categories[item] = [
                average - 2 * sd, average - 1 * sd, average, average + 1 * sd,
                average + 2 * sd
            ]

    #create meta- dictionary of categories
    variable2category = {}
    for thing in info_dict.keys():
        variable2category[thing] = {}
    try:
        for item in info_dict.keys():
            for jtem in keys:
                variable2category[item][jtem] = determine_category(
                    categories, jtem, info_dict[item][jtem], ranks)
    except:
        pass
    return variable2category


def determine_category(categories, variable, value, ranks):
    '''Determine category of data (extremely low, low, below average, above 
    average, high, extremely high'''
    if variable in ranks:
        compare = categories[variable]
        if np.isnan(value):
            return 'no data'
        if value < compare[0]:
            return 'extremely high'
        elif value < compare[1]:
            return 'high'
        elif value < compare[2]:
            return 'above average'
        elif value < compare[3]:
            return 'below average'
        elif value < compare[4]:
            return 'low'
        else:
            return 'extremely low'
    else:
        compare = categories[variable]
        if np.isnan(value):
            return 'no data'
        if value < compare[0]:
            return 'extremely low'
        elif value < compare[1]:
            return 'low'
        elif value < compare[2]:
            return 'below average'
        elif value < compare[3]:
            return 'above average'
        elif value < compare[4]:
            return 'high'
        else:
            return 'extremely high'


def list_countries(choices, trends):
    '''Return a list of countries who have specified categories of specificed
    variables'''
    possibilities = list(trends.keys())[:]
    for country in trends:
        for item in choices:
            if trends[country][item[0]] != item[1]:
                possibilities.remove(country)
                break
    return (possibilities)


def predict_happiness(happiness_level, keys, trends):
    '''Given a level of happiness, return the probability of any level of every
    variable'''
    predict = {}
    happy = list_countries([['Ladder', happiness_level]], trends)
    for item in keys:
        predict[item] = {
            'no data': 0,
            'extremely high': 0,
            'high': 0,
            'above average': 0,
            'below average': 0,
            'low': 0,
            'extremely low': 0
        }
    for country in happy:
        for item in keys:
            for jtem in predict[item].keys():
                if trends[country][item] == jtem:
                    predict[item][jtem] += 1

    for item in predict:
        for jtem in predict[item]:
            predict[item][jtem] = predict[item][jtem] / len(happy)
    return predict


#TODO: change to include all variables
def happy_probabilities(key, predict, title, ranks):
    '''Plot trends of countries with particular levels of happiness'''
#    if 'Military' not in key:
#        #TODO: this makes corruption look like it is going the wrong way. Reverse it with military
#        plt.bar(
#            list(range(len(list(predict[key].values())))[1:]),
#            list(predict[key].values())[1:])
#    else:
        #TODO: this is not correct   
    values = [predict[key]['extremely low'],predict[key]['low'],
              predict[key]['below average'],predict[key]['above average'],
              predict[key]['high'],predict[key]['extremely high']]
    plt.bar(
        list(range(len(values))),
        (values))
    plt.xticks(
        list(range(len(values))),
        ('extremely \nlow', 'low', 'below \naverage', 'above \naverage',
         'high', 'extremely \nhigh'))
    plt.title(title)
    plt.ylabel('Percent')
    plt.savefig('data/' + title ,  bbox_inches='tight')
    plt.show()


def remove_outliers(outliers, info):
    '''return copy of info_dict without particular entries in order to plot without 
    outliers'''
    info_copy = info.copy()
    for item in outliers:
        del info_copy[item]
    return info_copy


def plot_2var(info_dict, keys, happiness):
    '''plot two-variable relationships. Plot all variabes against all measures
    of happiness as well as "similar" variables against each other'''
    plot_relationships(info_dict, 'Percent Military(of GDP)',
                       'Percent Military(of Gov)', ranks)
    plot_relationships(info_dict, 'GDP', 'GDP 2', ranks)
    plot_relationships(info_dict, 'Pos Affect', 'Neg Affect', ranks)
    plot_relationships(info_dict, 'Pos Affect', 'Ladder', ranks)
    plot_relationships(info_dict, 'Ladder', 'Neg Affect', ranks)

    for item in keys:
        for jtem in happiness:
            if item == 'GDP 2':
                plot_relationships(info_dict, jtem, item, ranks)
            else:
                plot_relationships(info_dict, item, jtem, ranks)


#TODO: make this not hard coded...
def confusion_matrix():
    '''Create Confusion Matrix of correlation coefficients for all variables.
    If the p-value is below 0.05, mark correlation coefficient as 0.'''
    array = [[0, 0, 0], [0.818, 0.386, 0.622], [0.547, 0.681,
                                                0.430], [0.190, 0.210, 0],
             [0.5, 0.355, 0.338], [0.813, 0.305, 0.542], [0.1814, 0, 0], [
                 0.817, 0.33, 0.489
             ], [0, 0.245, 0], [0.214, 0.226, 0.216], [0.5884, 0, 0.3393],
             [0.5495, 0.3181, 0.4869], [0.6109, 0.22556, 0.5259], [0, 0, 0]]

    df_cm = pd.DataFrame(
        array,
        index=[
            '', "Social Support", "Freedom", 'Corruption', 'Generosity',
            'GDP Rank', 'GDP Raw', 'Life Expectancy', 'Military Spending- GDP',
            'Military Spending- Gov', 'Poverty Index',
            'Primary School Completion', 'Literacy', ''
        ],
        columns=['Ladder', 'Positive Affect', 'Negative Affect'])

    annotate = pd.DataFrame(
        [[' ', '', ''], [0.818, 0.386, 0.622], [0.547, 0.681, 0.430],
         [0.190, 0.210, 0], [0.5, 0.355, 0.338], [0.813, 0.305, 0.542], [
             0.1814, 0, 0
         ], [0.817, 0.33, 0.489], [0, 0.245, 0], [0.214, 0.226, 0.216],
         [0.588, 0, 0.339], [0.550, 0.318, 0.487], [0.611, 0.226,
                                                    0.526], ['', '', '']])
    plt.figure(figsize=(10, 7))
    plt.title('Correlation Coefficients')
    sn.heatmap(data=df_cm, annot=annotate, cmap='PuBu', cbar=True, fmt='')
    plt.savefig('data/confusion', bbox_inches='tight')


def get_info_dict():
    '''reads in the data and returns an info dict'''
    # read in data
    info_dict = read_happiness(keys + happiness)
    info_dict = read_military('data/milt_per_gov.csv', info_dict, TRANSLATE_DICT,
                              '(of Gov)', keys + happiness)
    info_dict = read_military('data/milt_GDP_per.csv', info_dict, TRANSLATE_DICT,
                              '(of GDP)', keys + happiness)
    info_dict = read_data('GDP 2', 'data/gdp_data.csv', TRANSLATE_DICT, info_dict, keys + happiness)
    info_dict = read_data('Poverty', 'data/poverty_data.csv', TRANSLATE_DICT,
                          info_dict, keys + happiness)
    info_dict = read_education(TRANSLATE_DICT, info_dict, keys + happiness)

    return info_dict


##############################################################################
'''Reference Info'''
if __name__ == "__main__":
    ##############################################################################
    '''Work'''

    info_dict = get_info_dict()

    #plot 2- Variable Scatterplots
    #plot_2var(info_dict, keys, happiness)

    #find trends
    #TODO: adjust for new data
    trends = find_trends(info_dict, keys, ranks)

    #
    for item in keys:
        predict = predict_happiness('extremely high', keys, trends)
        happy_probabilities(item, predict,
                            'Levels of ' + item + ' Among Happiest Countries', ranks)
        predict = predict_happiness('extremely low', keys, trends)
        happy_probabilities(item, predict,
                            'Levels of ' + item + ' Among Least Happy Countries', ranks)

    print(trends['Finland'])
    print(info_dict['Finland'])
    print(predict)
    #plot confusion matrix
    #confusion_matrix()


##############################################################################
