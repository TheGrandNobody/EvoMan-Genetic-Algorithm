# test boxplot function

# imports
import csv
import matplotlib.pyplot as plt

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
    boxplot('test.csv')