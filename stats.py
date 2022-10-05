import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#Start making the line plot over the different genarations 
#plt.style.use('_mpl-gallery')

#BubbleManneat_Stats = pd.read_csv("stats/BubbleManneatStatsFile.csv")
file1 = open("stats/BubbleManneatStatsFile.csv", "r")
BubbleManneat_Stats = file1.str.split(pat = ',', expand=False)

BubbleManneat_Stats_results = []
BubbleManneat_Stats_results = [range(1,15)]
# list = [[]]

for j in range (2,46,3):
    mean_mean_genaration = BubbleManneat_Stats[j].mean()
    stdev_mean_genaration = BubbleManneat_Stats[j].stdev()
    mean_max_genaration = BubbleManneat_Stats[j+1].max()
    stdev_max_genaration = BubbleManneat_Stats[j+1].stdev()
    BubbleManneat_Stats_results.append(mean_mean_genaration)
    BubbleManneat_Stats_results.append(stdev_mean_genaration)
    BubbleManneat_Stats_results.append(stdev_max_genaration)
    BubbleManneat_Stats_results.append(mean_max_genaration)

#Making line plot and adding mean and standaard deviation 
    line_plot_results = plt.plot (BubbleManneat_Stats_results[2])
    line_plot_results = plt.fill_between(range(len(BubbleManneat_Stats_results)),BubbleManneat_Stats_results[2]-BubbleManneat_Stats_results[3],BubbleManneat_Stats_results[2]+BubbleManneat_Stats_results[3],alpha=.1)
    line_plot_results = plt.plot (BubbleManneat_Stats_results[5])
    line_plot_results = plt.fill_between(range(len(BubbleManneat_Stats_results)),BubbleManneat_Stats_results[5]-BubbleManneat_Stats_results[4],BubbleManneat_Stats_results[5]+BubbleManneat_Stats_results[4],alpha=.1)

#Making boxplot of the best individuals that were tested 









