import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#Start making the line plot over the different genarations 
#plt.style.use('_mpl-gallery')
name = ["BubbleManneatStatsFile.csv", "BubbleManesneatStatsFile.csv", "CrashManneatStatsFile.csv", "CrashManesneatStatsFile.csv", "WoodManneatStatsFile.csv", "WoodManesneatStatsFile.csv"]


BubbleManneat_Stats = pd.read_csv("BubbleManneatStatsFile.csv")
BubbleManneat_Stats = BubbleManneat_Stats.str.split(',', expand=True)

BubbleManneat_Stats_results = []
BubbleManneat_Stats_results = [range(1,15)]


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









