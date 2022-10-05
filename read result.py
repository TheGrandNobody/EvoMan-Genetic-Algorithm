from glob import glob
import pandas as pd
import numpy as np

paths = glob(r"train\*\results.txt")

df = pd.DataFrame()
for i in paths:
    df_1 = pd.read_csv(i, sep=" ")
    df = df.append(df_1, ignore_index=True)

best_1 = df["best"].values
mean_1 = df["mean"].values

output = []
for elem in best_1:
    if elem != "best":
        output.append(elem)
best_1 = np.array(output)

output = []
for elem in mean_1:
    if elem != "mean":
        output.append(elem)
mean_1 = np.array(output)

best = best_1.reshape(len(paths), -1)
mean = mean_1.reshape(len(paths), -1)

mean_best = []
for row in range(len(best)):
    test = []
    for col in range(len(best[0])):
        test.append(col)
        test.append(mean[row][col])
        test.append(best[row][col])

    mean_best.append(test)

new_df_2 = pd.DataFrame(columns=range(len(mean_best[0])), index=range(len(mean_best)))
for i in range(len(mean_best)):
    new_df_2.loc[i] = mean_best[i]

new_df_2.to_csv('train/for_train_new_1.csv')
