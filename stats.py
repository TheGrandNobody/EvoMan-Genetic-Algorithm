import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Import the statistiek files 
BubbleManStatsFile = pd.read_csv('BubbleManStatsFile.csv')
BubbleManStats = BubbleManStatsFile.str.split(',', expand=True)

result_BubbleMan = []

for i in range (2,75,3):
    mean_genaration = BubbleManStats[i].mean()
    max_genaration = BubbleManStats[i+1].max()
    result_BubbleMan.append(mean_genaration)


for i in range(3,75,3):
    max_genaration = BubbleManStats[i].max()
    result_BubbleMan.append(max_genaration)









