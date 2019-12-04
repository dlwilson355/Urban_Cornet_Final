#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:10:49 2019

@author: sarafergus
"""
#going to have to change this
from WHR_EDA import get_var
from WHR_EDA import info_dict
import matplotlib.pyplot as plt
import statistics
    
##############################################################################

def histogram(information, title, bins):
    '''Make histogram of military measurements'''
    fig,ax = plt.subplots()
    if title != 'GDP 2':
        textstr = 'Mean: ' + str(round(statistics.mean(information),2)) + '%\n SD: ' + str(round(statistics.stdev(information),4))
    else:
        textstr = 'Mean: ' + '551 Billion' + '\n SD: ' + '2 Trillion'
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    ax.text(0.67, 0.95, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', bbox=props)
    plt.hist(information, bins = bins)
    plt.title(title)
    plt.xlabel('Percent')
    plt.ylabel('Count')
    plt.savefig('./hist' + title, bbox_inches='tight')
    plt.show()

def cdf(information, title, marks, vals):
    '''make CDF of military measurements, marking countries of interest'''
    x_cdf = sorted(information)
    N = len(x_cdf)
    y_cdf = [ (N-1.0-i)/N for i in range(N) ]
    plt.title(title)
    plt.plot(x_cdf, y_cdf, '.-')
    for i in range(len(vals)):
        plt.plot([vals[i], vals[i]], [0,1], '--', label=marks[i]) 
    plt.xlabel("Spending Percent$(x)$");plt.ylabel("$P(X>x)$"); plt.legend();
    plt.savefig('./cdf' + title, bbox_inches='tight')
    plt.show()
    
##############################################################################
    
marks = ['United States', 'Finland', 'Central African Republic', 'Paraguay', 'Turkey', 'Taiwan']
    
milt_gov, _ = get_var(info_dict, 'Percent Military(of Gov)', [])
milt_gov = [x for x in milt_gov if x != 0]
milt_gdp, _ = get_var(info_dict, 'Percent Military(of GDP)', [])
milt_gdp = [x for x in milt_gdp if x != 0]
poverty, _ = get_var(info_dict, 'Poverty', [])
poverty = [x for x in poverty if x != 0]
gdp, _ = get_var(info_dict, 'GDP 2', [])
gdp = [x for x in gdp if x != 0]


vals = [info_dict[item]['Percent Military(of Gov)'] for item in marks if info_dict[item]['Percent Military(of Gov)']!= 0]  
vals2 = [info_dict[item]['Percent Military(of GDP)'] for item in marks if info_dict[item]['Percent Military(of GDP)']!= 0]  
vals3 = [info_dict[item]['Poverty'] for item in marks if info_dict[item]['Poverty']!= 0]  
vals4 = [info_dict[item]['GDP 2'] for item in marks if info_dict[item]['GDP 2']!= 0]  



histogram(milt_gov, 'Military Spending as a Percent of\n Government Spending', 20)
histogram(milt_gdp, 'Military Spending as a Percent of GDP', 20)
histogram(poverty, 'Poverty', 30)
histogram(gdp, 'GDP 2', 20)

cdf(milt_gov, "Military Spending by Percent of\n Government Spending", marks, vals)
cdf(milt_gdp, "Military Spending by Percent of GDP", marks, vals2)
cdf(poverty, "Poverty", marks, vals3)
cdf(gdp, "GDP 2", marks, vals4)

