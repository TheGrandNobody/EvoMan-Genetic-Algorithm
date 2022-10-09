import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import numpy as np
import sys
import csv

def line_plot(all_max_gens, all_mean_gens, max_std_lower, mean_std_lower, max_std_upper, mean_std_upper, names):
    for n in range(2):
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.set_title(sys.argv[n+1] + " line plot of best and mean performance",size=10)
        plt.legend(loc="best",prop={'size': 5})
        plt.set_xlim(0, 14)
        plt.set_ylim(-10, 100)
        for i in range(2):
            plt.plot(all_max_gens[i+n], color=None, alpha=1.0,label="Best "+names[i+n])
            plt.plot(all_mean_gens[i+n], color=None, alpha=1.0,label="Mean "+names[i+n])
            plt.fill_between(x=range(15), y1=max_std_lower[i+n],y2=max_std_upper[i+n], alpha=.2)
            plt.fill_between(x=range(15), y1=mean_std_lower[i+n],y2=mean_std_upper[i+n], alpha=.2)
        plt.savefig(sys.argv[n+1])
        plt.clf()

def boxplot(data_file):
    # initialize all_data and all_groups
    all_data = []
    all_groups = []
    # read csv file
    with open(data_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            all_groups.append(str(row[0] + row [1]))
            data_1box = []
            for i in range(3, len(row)-1, 2):
                data_1box.append(float(row[i]))
            all_data.append(data_1box)

    # boxplot
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(all_data, patch_artist=True)
    
    # settings
    plt.title("2 EAs, 2 pairs of enemies")
    ax.set_xticklabels(all_groups)
    colors = ['gray', 'red', 'green', 'blue', 'cyan']

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    for median in bp['medians']:
        median.set(color ='black',
                    linewidth = 1.5,
                    linestyle ="-")
    # save plot  
    plt.savefig("Boxplot")
    return 
  
if __name__ == "__main__":
    # Gather paths
    paths = glob(r"stats/*")
    # Initialize std lists
    std_mean_gens1=[]
    std_max_gens1=[]
    # Initialize mean/max lists
    all_mean_gens1=[]
    all_max_gens1=[]
    # Initialize upper/lower std lists
    mean_std_upper1 = []
    mean_std_lower1 = []
    max_std_upper1 = []
    max_std_lower1 = []
    # Initialize a list of names
    names=[]

    for i in paths:
        name = (i.split("/"))[1]
        if "esneat" in name:
            names.append("ES-HyperNEAT")
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
    line_plot(all_max_gens1, all_mean_gens1, max_std_lower1, mean_std_lower1, max_std_upper1, mean_std_upper1, names)
    