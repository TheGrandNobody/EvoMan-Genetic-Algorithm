# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import pandas as pd

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os


num_all_enemies=2
test_five_times=2
num_of_folders=3

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

n_hidden_neurons = 10

experiment_name = 'train/train_1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes simulation in multi evolution mode, for multiple static enemies.
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  multiplemode="no",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'test' # train or test

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(x):
    temp = []
    for i in x:
        fit = simulation(env, i)
        temp.append(fit)
    return np.array(temp)


if __name__ == "__main__":
    means=[]
    mean_of_eight_enemy = []
    experiment_name_for_save=[]

    for f in range(num_of_folders):
        experiment_name_for_save.append("train_"+str(f+1))
        mean_dict = {}
        for i in range(1,num_all_enemies+1):
            temp = []
            temp_1 = []
            temp_2 = []
            for j in range(test_five_times):
                # loads file with the best solution for testing
                experiment_name = 'train/train_' + str(f+ 1)
                env.update_parameter('experiment_name', experiment_name)
                env.update_parameter('enemies', [i])
                if run_mode =='test':
                    bsol = np.loadtxt(experiment_name+'/best.txt')
                    print( '\n RUNNING SAVED BEST SOLUTION \n')
                    # env.update_parameter('speed','normal')
                    fitness = evaluate([bsol])
                    temp.append(fitness)
                temp_1.append(env.get_enemylife())
                temp_2.append(env.get_playerlife())
            mean = np.mean(temp)
            enemy_life_mean=np.mean(temp_1)
            player_life_mean=np.mean(temp_2)

            mean_dict["enemy_life_mean"]=enemy_life_mean
            mean_dict["player_life_mean"]=player_life_mean
            mean_dict["enemie_num " +str(i)]=mean

        means.append(mean_dict)

    # calculate the mean of the means
    for i in means:
        mean_of_eight_enemy.append(sum(i.values())/len(i))

    df = pd.DataFrame(means)
    df["The mean of the means"]=mean_of_eight_enemy
    df.index=experiment_name_for_save
    For_test = 'For test'
    if not os.path.exists(For_test):
        os.makedirs(For_test)
    df.to_csv(For_test+'/export_dataframe.csv')

    best_of_bests=np.argmax(mean_of_eight_enemy)
    print(f"The best of bests is train_{best_of_bests+1}")

    player_life=[]
    enemy_life=[]



    sys.exit(0)
