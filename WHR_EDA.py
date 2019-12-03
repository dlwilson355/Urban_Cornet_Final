#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 20:25:27 2019
@author: sarafergus
"""

import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import statistics

def get_var(info_dict, key, marks):
  temp = []
  mark_vals = {}
  for country, values in info_dict.items():
    temp.append(values[key])
    if country in marks:
        mark_vals[country] = values[key]
  return temp, mark_vals

def read_happiness():
    i = 0
    info_dict = {}
    f = open('./WHR.csv')
    for line in f:
        i = i + 1
        if i > 3:
            line = line.strip().split(',')
            try:
                info_dict[line[0]] = {'Ladder': int(line[1]), 'Pos Affect': int(line[3]), 
                     'Neg Affect': int(line[4]), 'Social Support': int(line[5]), 
                     'Freedom': int(line[6]), 'Corruption': int(line[7]), 'Generosity': int(line[8]), 
                     'GDP': int(line[9]), 'Life Exp': int(line[10])}
            except:
                #no data. Make zero. Do not include in statistics and charts (piecewise)
                for i in range(len(line)):
                    if len(line[i]) < 1:
                        line[i] = 0
                info_dict[line[0]] = {'Ladder': int(line[1]), 'Pos Affect': int(line[3]), 
                     'Neg Affect': int(line[4]), 'Social Support': int(line[5]), 
                     'Freedom': int(line[6]), 'Corruption': int(line[7]), 'Generosity': int(line[8]), 
                     'GDP': int(line[9]), 'Life Exp': int(line[10])}
    return info_dict


#TODO: delete pairwise before plotting/doing relationship statistics.
def pairwise_delete(x, y):
    x_copy = x[:]
    y_copy = y[:]
    for i in range(len(x)):
        if x[i] == 0 or y[i] == 0:
            x_copy.remove(x[i])
            y_copy.remove(y[i])
        #for Saudi Arabia Outlier
#        if x[i] > 6.0:
#            x_copy.remove(x[i])
#            y_copy.remove(y[i])
    return x_copy, y_copy

#TODO: reverse axes? 
#TODO: mark particular countries, if possible
def plot_relationships(info_dict, x_key, y_key):
    ranks = ['Pos Affect','Neg Affect','Ladder', 'Social Support','Freedom','Corruption', 
            'Generosity','GDP','Life Exp']
    marks = ['United States', 'Finland', 'Central African Republic', 'Paraguay', 'Turkey', 'Taiwan']
    x, mark_x = get_var(info_dict, x_key, marks)
    y, mark_y = get_var(info_dict, y_key, marks)
    x, y = pairwise_delete(x, y)
    #linear regression
    m , b, r, p, sd = scipy.stats.linregress(x, y)
    line = [m*x1 + b for x1 in x]
    fig,ax = plt.subplots()
    textstr = 'r-value: ' + str(round(r,4)) + '\n p-value: ' + str(round(p,4))
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', bbox=props)
    if x_key in ranks:
        ax.set_xlim(max(x) + 10,min(x) - 10)
        plt.xlabel(x_key + ' Ranking')
    else:
        plt.xlabel(x_key)
    if y_key in ranks:
        ax.set_ylim(max(y) + 10, min(y) -10) 
        plt.ylabel(y_key + ' Ranking')
    else:
        plt.ylabel(y_key)
    plt.scatter(x,y)
    plt.scatter(list(mark_x.values()), list(mark_y.values()))
    if 'Military'in x_key or 'Military' in y_key:
#        print(mark_x, mark_y)
        pass
    for value in mark_x.keys():
        if mark_x[value]!= 0 and mark_y[value]!=0:
            props = dict(boxstyle='round', facecolor='white', alpha=0.75)
            ax.annotate(value, (mark_x[value]-5, mark_y[value]-5), bbox = props)
    plt.title(x_key + ' versus ' + y_key)
    plt.plot(x, line, 'r')
    plt.show()
#    plt.savefig('./EDA' + x_key + y_key)
    return m
    
def monte_carlo():
    #monte carlo permutation test
    list_r_slopes = []
    vals = list(range(1,157))
    for _ in range(1000):
        yr = vals.copy()
        np.random.shuffle(yr)
        slope_r = scipy.stats.linregress(vals,yr)[0]
        list_r_slopes.append(slope_r)
    return list_r_slopes

def p_hat(slopes, slope):
    return(2.0*len([si for si in slopes if si >= slope]) / len(slopes))

#TODO: deal with commas in the Country name, EX congo
#TODO: combine with other read function
def read_military(filename, info_dict, translate, percent_of):
    f = open(filename)
    these_countries= []
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
        
    #missing countries
        these_countries.append(country)
    diff = set(these_countries)-set(list(info_dict.keys()))
    for item in diff:
        if item not in these_countries:
#            print(item)
            pass
            
    return info_dict
    

#TODO: incorpoate this into checks 
def reconcile(info_dict):
    for item in info_dict.keys():
        for jtem in info_dict['Colombia'].keys():
            if jtem not in info_dict[item].keys():
                info_dict[item][jtem] = 0
    return info_dict
        
#def bad_data
    #pass
def find_trends(info_dict):
    #break into categories (extremely low, low, below average, above average, high, extremely high)
    ranks = ['Pos Affect','Neg Affect','Ladder', 'Social Support','Freedom','Corruption', 
            'Generosity','GDP','Life Exp']
    keys = ['Pos Affect','Neg Affect','Ladder', 'Social Support','Freedom','Corruption', 
            'Generosity','GDP','Life Exp','Percent Military(of GDP)',
            'Percent Military(of Gov)']
    data = {}
    categories = {}
    for val in keys:
        info, _ = get_var(info_dict, val, [])
        data[val] = info
    for item in data.keys():
        if item in ranks:
            categories[item] = [(len(data[item]))/6, (len(data[item]))/3, len(data[item])/2, (2*len(data[item]))/3, (5*len(data[item]))/6]
        else:
            average = statistics.mean(data[item])
            sd = statistics.stdev(data[item])
            categories[item] = [average- 2*sd, average - 1*sd, average, average + 1*sd, average + 2*sd]
    
    #create meta- dictionary of categories
    variable2category = {}
    for thing in info_dict.keys():
        variable2category[thing] = {}
    try:
        for item in info_dict.keys():
            for jtem in info_dict['Colombia'].keys():
                variable2category[item][jtem] = determine_category(categories, jtem, info_dict[item][jtem])
    except:
        print(item, jtem)
        print(info_dict[item][jtem])
        print(determine_category(categories, jtem, 1))
        variable2category[item][jtem] = 'hi'
        print(variable2category)
    return variable2category       
            
def determine_category(categories, variable, value):
    compare = categories[variable]
    if value == 0:
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
    possibilities = list(trends.keys())[:]
    for country in trends:
        for item in choices:
            if trends[country][item[0]] != item[1]:
                possibilities.remove(country)
                break
    return(possibilities)
    
translate_dict = {'Lao PDR':'Laos','Venezuela, RB':'Venezuela',
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
##############################################################################

info_dict = read_happiness()


info_dict = read_military('./milt_per_gov.csv', info_dict, translate_dict, '(of Gov)')
#print(info_dict['Colombia'])
info_dict = reconcile(info_dict)
#print(info_dict)
info_dict = read_military('./milt_GDP_per.csv', info_dict, translate_dict, '(of GDP)')
#print(info_dict['Colombia'])
info_dict = reconcile(info_dict)

plot_relationships(info_dict, 'Percent Military(of GDP)', 'Percent Military(of Gov)')

for item in info_dict.keys():
    if info_dict[item]['Percent Military(of GDP)'] > 6.0:
#        print(item)
        pass
        

keys = ['Pos Affect','Neg Affect','Ladder', 'Social Support','Freedom','Corruption', 
            'Generosity','GDP','Life Exp','Percent Military(of GDP)',
            'Percent Military(of Gov)']

done = []
#mc = monte_carlo()
for i in list(keys):
    for j in list(keys): 
        if i != j and set([i,j]) not in done:
            m = plot_relationships(info_dict, i, j)
#            print(p_hat(mc, m))
            done.append(set([i,j]))
        


milt_gov, _ = get_var(info_dict, 'Percent Military(of Gov)', [])
milt_gov = [x for x in milt_gov if x != 0]

marks = ['United States', 'Finland', 'Central African Republic', 'Paraguay', 'Turkey', 'Taiwan']

vals = [info_dict[item]['Percent Military(of Gov)'] for item in marks if info_dict[item]['Percent Military(of Gov)']!= 0]  

fig,ax = plt.subplots()
textstr = 'Mean: ' + str(round(statistics.mean(milt_gov),2)) + '%\n SD: ' + str(round(statistics.stdev(milt_gov),4))
props = dict(boxstyle='round', facecolor='white', alpha=1)
ax.text(0.67, 0.95, textstr, transform=ax.transAxes, fontsize=14,
verticalalignment='top', bbox=props)
#plt.hist(milt_gov, bins = 20)
 
plt.title('Military Spending as a Percent of Government Spending')
#add legend
plt.xlabel('Percent')
plt.ylabel('Count')
#plt.show()
milt_gdp, _ = get_var(info_dict, 'Percent Military(of GDP)', [])
vals2 = [info_dict[item]['Percent Military(of GDP)'] for item in marks if info_dict[item]['Percent Military(of GDP)']!= 0]  
milt_gdp = [x for x in milt_gdp if x != 0]
fig,ax = plt.subplots()
textstr = 'Mean: ' + str(round(statistics.mean(milt_gdp),2))+ '% \n SD: ' + str(round(statistics.stdev(milt_gdp),2))
props = dict(boxstyle='round', facecolor='white', alpha=1)
ax.text(0.68, 0.95, textstr, transform=ax.transAxes, fontsize=14,
verticalalignment='top', bbox=props)
#plt.hist(milt_gdp, bins = 'auto')
 
plt.title('Military Spending as a Percent of GDP')
plt.xlabel('Percent')
plt.ylabel('Count')
plt.show()

#print(info_dict['United States'])
#

print(info_dict['Sweden'])

#print(list(info_dict.keys()))

x_cdf = sorted(milt_gov)
N = len(x_cdf)
y_cdf = [ (N-1.0-i)/N for i in range(N) ]
plt.title("Military Spending by Percent of Government Spending")
plt.plot(x_cdf, y_cdf, '.-')
for i in range(len(vals)):
    plt.plot([vals[i], vals[i]], [0,1], '--', label=marks[i]) 
plt.xlabel("Spending Percent$(x)$");plt.ylabel("$P(X>x)$"); plt.legend();
plt.show()

x_cdf = sorted(milt_gdp)
N = len(x_cdf)
y_cdf = [ (N-1.0-i)/N for i in range(N) ]
plt.title("Military Spending by Percent of GDP")
plt.plot(x_cdf, y_cdf, '.-')
for j in range(len(vals2)):
    plt.plot( [vals2[j], vals2[j]], [0,1], '--', label=marks[j] ) 
plt.xlabel("Spending Percent $(x)$");plt.ylabel("$P(X>x)$"); plt.legend();
plt.show()


trends = find_trends(info_dict)
#print(list_countries([['Corruption','extremely low'], ['GDP','extremely low'], ['Freedom','extremely low'],['Life Exp','extremely low']],trends))

cool_countries = list_countries([['Corruption','extremely low'], ['GDP','extremely low'], ['Freedom','extremely low'],['Life Exp','extremely low'], ['Generosity','extremely low'], ['Social Support', 'extremely low'], ['Percent Military(of GDP)','above average' ]],trends)
#for item in cool_countries:
#    print(trends[item]['Ladder'])
#print(cool_countries)

#TODO: among extremely happy countries, what are the probabilities of each?
def predict_happiness(happiness_level, keys, trends): 
    predict = {}
    happy = list_countries([['Ladder',happiness_level]], trends)
    for item in keys:
        predict[item] = {'no data':0 , 'extremely high': 0 , 'high': 0 ,'above average': 0 , 'below average' : 0, 'low': 0 , 'extremely low': 0}
    for country in happy:
        for item in keys:
            for jtem in predict[item].keys():
                if trends[country][item] == jtem:
                    predict[item][jtem] += 1
    
    for item in predict:
        for jtem in predict[item]:
            predict[item][jtem] = predict[item][jtem]/len(happy)
    return predict
        
def happy_probabilities(key, predict, title):
    if 'Military' not in key:
        #TODO: this makes corruption look like it is going the wrong way. Reverse it with military
        plt.bar(list(range(len(list(predict[key].values())))[1:]), list(predict[key].values())[1:])
    else:
        #TODO: this is not correct
        plt.bar(list(range(len(list(predict[key].values())))[1:]), (list(predict[key].values())[1:]))
    plt.xticks(list(range(len(list(predict[key].values())))[1:]), ('extremely \nlow', 'low', 'below \naverage', 'above \naverage','high','extremely \nhigh'))
    plt.title(title) 
    plt.ylabel('Percent')
    plt.show()

for item in keys:    
    predict = predict_happiness('extremely low', keys, trends)
    happy_probabilities(item, predict, 'Levels of ' + item + ' Among Happiest Countries')

for item in keys:    
    predict = predict_happiness('extremely high', keys, trends)
    happy_probabilities(item, predict, 'Levels of ' + item + ' Among Least Happy Countries')

from wealth_data import load_data
new_gdp_data = load_data('./gdp_data.csv') 

for item in new_gdp_data.keys():
    if item in translate_dict.keys():
        jtem = translate_dict[item]
    else:
        jtem = item
    try:
        info_dict[jtem]['GDP 2'] = new_gdp_data[item]
    except:
        pass

#make this a function
poverty_data = load_data('./poverty_data.csv')

for item in poverty_data.keys():
    if item in translate_dict.keys():
        jtem = translate_dict[item]
    else:
        jtem = item
    try:
        info_dict[jtem]['Poverty'] = poverty_data[item]
    except:
        pass

from load_education import load_data_education
variables = ["Primary completion rate, total (% of relevant age group)","Literacy rate, adult total (% of people ages 15 and above)"]
education_data = load_data_education('./education.csv',variables)

for item in education_data.keys():
    if item in translate_dict.keys():
        jtem = translate_dict[item]
    else:
        jtem = item
    try:
        info_dict[jtem]['Primary Completion'] = float(education_data[item][2])
        info_dict[jtem]['Literacy'] = float(education_data[item][5])
        print(info_dict[jtem])
    except:
        pass

#print(info_dict['United States'])
#check 
translated = [translate_dict[item] for item in new_gdp_data.keys() if item in translate_dict.keys()] + [item for item in new_gdp_data.keys() if item not in translate_dict.keys()]
for item in info_dict.keys():
    if item not in translated:
        print(item)

info_dict = reconcile(info_dict)  

for item in ['GDP', 'Ladder', 'Pos Affect', 'Neg Affect']:
    plot_relationships(info_dict, item, 'GDP 2')
    
info_dict_copy = info_dict.copy()
del info_dict_copy['United States']
del info_dict_copy['China']
print(info_dict_copy == info_dict)

for item in ['GDP', 'Ladder', 'Pos Affect', 'Neg Affect']:
    plot_relationships(info_dict_copy, item, 'GDP 2')
  
for item in ['Ladder', 'Pos Affect', 'Neg Affect']:
    plot_relationships(info_dict, item, 'Poverty')
    
for item in ['Ladder', 'Pos Affect', 'Neg Affect']:
    plot_relationships(info_dict, item, 'Primary Completion')
    
for item in ['Ladder', 'Pos Affect', 'Neg Affect', 'Primary Completion']:
    plot_relationships(info_dict, item, 'Literacy')
#
#
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = [[0,0,0],
         [0.818, 0.386, 0.622],
         [0.547,0.681,0.430],
         [0.190, 0.210,0],
         [0.5,0.355,0.338],
         [0.813,0.305,0.542],
         [0.1814,0,0],
         [0.817,0.33,0.489],
         [0,0.245,0],
         [0.214,0.226,0.216],
         [0.5884, 0, 0.3393],
         [0.5495, 0.3181,0.4869],
         [0.6109, 0.22556, 0.5259],
         [0,0,0]]
df_cm = pd.DataFrame(array, index = ['',"Social Support", "Freedom", 'Corruption', 'Generosity', 'GDP Rank','GDP Raw','Life Expectancy','Military Spending- GDP','Military Spending- Gov','Poverty Index','Primary School Completion','Literacy',''],
                  columns = ['Ladder','Positive Affect','Negative Affect'])
#
#array2 = [
#         [0.818, 0.386, 0.622],
#         [0.547,0.681,0.430],
#         [0.190, 0.210,0],
#         [0.5,0.355,0.338],
#         [0.813,0.305,0.542],
#         [0.1814,0,0],
#         [0.817,0.33,0.489],
#         [0,0.245,0],
#         [0.214,0.226,0.216],
#         ]
#df_cm2 = pd.DataFrame(array2, index = ['',"Social Support", "Freedom", 'Corruption', 'Generosity', 'GDP Rank','GDP Raw','Life Expectancy','Military Spending- GDP','Military Spending- Gov',''],
#                  columns = ['Ladder','Positive Affect','Negative Affect'])
#annotate = (np.asarray(["{0}\n{1:.2f}".format(text,data) for text, data in zip(text.flatten(), data.flatten())])).reshape(4,6)

annotate = pd.DataFrame([[' ','',''],
         [0.818, 0.386, 0.622],
         [0.547,0.681,0.430],
         [0.190, 0.210,0],
         [0.5,0.355,0.338],
         [0.813,0.305,0.542],
         [0.1814,0,0],
         [0.817,0.33,0.489],
         [0,0.245,0],
         [0.214,0.226,0.216],
         [0.588, 0, 0.339],
         [0.550, 0.318,0.487],
         [0.611, 0.226, 0.526],
         ['', '', '']])
plt.figure(figsize = (10,7))

#TODO: sort this
sn.heatmap(data=df_cm, annot=annotate, cmap = 'PuBu', cbar = True, fmt='')
#sn.heatmap(df_cm2, annot=False, cmap= 'PuBu')
