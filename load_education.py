#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:00:10 2019
@author: noahstrac
"""

#download dataset:
#http://api.worldbank.org/v2/en/topic/4?downloadformat=csv

"""This function uses the dataset above and loads information on every country
in the dataset depending on which variables are specified. The returned dataset
is a dictionary in the form of "'country' : ['variable1', year1, 'stat1', 
'variable2', year2, 'stat2', ... 'variableN', yearN, 'statN']"
Note 1: stat and year are returned as strings due to having the possibility of 
having the value 'No Data', which denotes missing data for that variable of 
that country.
Note 2: This function accounts for 1 comma in the variable name. If there is
not exactly 1 comma in the variable name, this function will be off by +/- 1
year accordingly. Nearly every variable in the dataset has exactly 1 comma.
"""
def load_data_education(filename, variables):
    #initialize data dictionary
    data = dict()
    #initialize countries to ignore
    countries_to_ignore = ["World",
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
                           "Arab World"]
    #open file
    file = open(filename, 'r')
    #get data
    for line in file:
        #loop through variables given
        for variable in variables:
            if variable in line:
                #split csv line
                info = line.split(',')
                #get country
                country = info[0].strip('"')
                #determine if initializing a new dictionary country or if it's
                #already initialized
                if country in data:
                    #initialize variables
                    stat = ""
                    #years start at 1960, but there are 5 points in info
                    #before the years actually start. 1955 accounts for this
                    year_on = 1955
                    for i in info:
                        #if info in i is not empty
                        if i != "" and i != "\n" and i != '""':
                            #take out random quotes and set newest stat
                            stat = i.strip('"')
                            #set most recent year of a data point
                            year = year_on
                        year_on += 1
                    #info[3] is an indicator code and is the last guaranteed
                    #non-empty data point before the years start, so if there
                    #is no data, this value will be the value for stat and it
                    #always includes an S to start it
                    if "S" in stat:
                        stat = "No Data"
                        year = "No Data"
                    #add info to the country's data
                    data[country] += [variable, str(year), stat]
                else:
                    stat = ""
                    year_on = 1955
                    for i in info:
                        if i != "" and i != "\n" and i != '""':
                            stat = i.strip('"')
                            year = year_on
                        year_on += 1
                    if "S" in stat:
                        stat = "No Data"
                        year = "No Data"
                    #initialize country's first variable and data associated
                    if country not in countries_to_ignore:
                        data[country] = [variable, str(year), stat]
    file.close()
    return data

variables = ["Primary completion rate, total (% of relevant age group)","Literacy rate, adult total (% of people ages 15 and above)"]
print(load_data_education("education.csv",variables))
                
                
