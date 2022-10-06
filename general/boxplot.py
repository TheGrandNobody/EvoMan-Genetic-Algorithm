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
            row_list = []
            all_groups.append(row[0])
            for i in range(2, len(row)-1, 2):
                print(row[i])
                row_list.append(float(row[i])) # change for test.csv 
            all_data.append(row_list)

    # boxplot
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(all_data, patch_artist=True)
    
    # settings
    plt.title("2 EAs, 2 pairs of enemies")
    ax.set_xticklabels(all_groups)
    colors = ['black', 'red', 'green', 'blue', 'cyan']

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