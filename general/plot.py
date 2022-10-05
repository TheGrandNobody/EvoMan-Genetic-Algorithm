import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import numpy as np
import sys

paths = glob(r"stats/*")

std_mean_gens1=[]
std_max_gens1=[]

all_mean_gens1=[]
all_max_gens1=[]

mean_std_upper1 = []
mean_std_lower1 = []

max_std_upper1 = []
max_std_lower1 = []
names=[]

for i in paths:
    name = (i.split("/"))[1]
    if "simple" in name:
        names.append("Regular")
    else:
        names.append("NEAT")

    df = pd.read_csv(i,names = range(45))
    df.insert(loc=0,column="A",value=[0]*10)
    mean_gens=[]
    max_gens=[]

    for i in range(10):
        temp=[]
        temp_1=[]
        for j in range(0,45,3):
            temp.append(df[j][i])
            temp_1.append(df[j+1][i])
        mean_gens.append(temp)
        max_gens.append(temp_1)

    all_mean_gens = np.array(mean_gens)
    all_max_gens = np.array(max_gens)

    std_mean_gens1.append(np.std(all_mean_gens, axis=0))
    std_max_gens1.append(np.std(all_max_gens, axis=0))
    all_mean_gens1.append(np.mean(all_mean_gens, axis=0))
    all_max_gens1.append(np.mean(all_max_gens, axis=0))

    mean_std_upper1.append(np.mean(all_mean_gens, axis=0) + np.std(all_mean_gens, axis=0))
    mean_std_lower1.append(np.mean(all_mean_gens, axis=0) - np.std(all_mean_gens, axis=0))

    max_std_upper1.append(np.mean(all_max_gens, axis=0) + np.std(all_max_gens, axis=0))
    max_std_lower1.append(np.mean(all_max_gens, axis=0) - np.std(all_max_gens, axis=0))

for n in range(2):
    for i in range(2):
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.plot(all_max_gens1[i+n], color=None, alpha=1.0,label="Best "+names[i+n])
        plt.plot(all_mean_gens1[i+n], color=None, alpha=1.0,label="Mean "+names[i+n])
        plt.fill_between(x=range(15),y1=max_std_lower1[i+n],y2=max_std_upper1[i+n], alpha=.2)
        plt.fill_between(x=range(15),y1=mean_std_lower1[i+n],y2=mean_std_upper1[i+n], alpha=.2)
        plt.set_title(sys.argv[1] + " line plot of best and mean performance",size=10)
        plt.legend(loc="best",prop={'size': 5})
        plt.set_xlim(0, 14)
        plt.set_ylim(-10, 100)
        plt.savefig(sys.argv[1])
        plt.clf()