# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from controllers import simple
import pandas as pd

# imports other libs
import time
import numpy as np
import os


# num_all_enemies=2
num_all_enemies=[7,8]
test_five_times=2
num_of_folders=2

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
                  enemies=[num_all_enemies[0]],
                  multiplemode="no",
                  playermode="ai",
                  player_controller=simple(n_hidden_neurons),
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
    means = []
    mean_of_eight_enemy = []
    experiment_name_for_save = []

    enemy_life_mean = []
    player_life_mean = []

    for f in range(num_of_folders):
        mean_dict = {}
        enemy_life_mean_temp = {}
        player_life_mean_temp = {}

        experiment_name_for_save.append("train_" + str(f + 1))
        for i in num_all_enemies:
            temp = []
            temp_1 = []
            temp_2 = []
            for j in range(test_five_times):
                # loads file with the best solution for testing
                experiment_name = 'train/train_' + str(f + 1)
                env.update_parameter('experiment_name', experiment_name)
                env.update_parameter('enemies', [i])
                if run_mode == 'test':
                    bsol = np.loadtxt(experiment_name + '/best.txt')
                    print('\n RUNNING SAVED BEST SOLUTION \n')
                    # env.update_parameter('speed', 'normal')
                    fitness = evaluate([bsol])
                    temp.append(fitness)
                temp_1.append(env.get_enemylife())
                temp_2.append(env.get_playerlife())

            mean_dict["enemie_num " + str(i)] = np.mean(temp)
            enemy_life_mean_temp["enemy_life_mean"]=np.mean(temp_1)
            player_life_mean_temp["player_life_mean"]=np.mean(temp_2)
        means.append(mean_dict)
        enemy_life_mean.append(enemy_life_mean_temp)
        player_life_mean.append(player_life_mean_temp)

    # calculate the mean of the means
    for i in means:
        mean_of_eight_enemy.append(sum(i.values()) / len(i))

    df = pd.DataFrame(means)
    df2 = pd.DataFrame(enemy_life_mean)
    df1 = pd.DataFrame(player_life_mean)
    df = pd.concat([df,df1,df2],axis=1)

    df.index = experiment_name_for_save
    For_test = 'For test'
    if not os.path.exists(For_test):
        os.makedirs(For_test)
    df.to_csv(For_test + '/export_dataframe.csv')
    best_of_bests = np.argmax(mean_of_eight_enemy)
    print(f"The best of bests is train_{best_of_bests + 1}")

    sys.exit(0)

