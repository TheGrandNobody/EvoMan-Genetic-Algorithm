# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from controllers import simple

# imports other libs
import time
import numpy as np
import os


experiment_name = 'train/train_1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

n_hidden_neurons = 10

# initializes simulation in multi evolution mode, for multiple static enemies.
env = Environment(experiment_name=experiment_name,
                  enemies=[7,8],
                  multiplemode="yes",
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

run_mode = 'train' # train or test


# number of weights for multilayer with 10 hidden neurons.
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

dom_u = 1
dom_l = -1
npop = 10
gens = 6
offspring_rate = 0.25
mutation_rate = 0.25
last_best = 0

np.random.seed(400)

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

# parent_selection
def parent_selection(pop,fitness):
    offspring_amount= int(len(pop)*offspring_rate)
    return pop[::][:offspring_amount],fitness[::][:offspring_amount]

# crossover
def crossover(pop,fit):
    for i in range(len(pop[0])):
        np.random.shuffle(pop[::,i])
    return pop

def mutation(children):
    select_row= int(len(children)*mutation_rate)
    rendom_row = np.random.randint(0,len(children),size=select_row)
    select_col = int(len(children[0]) * mutation_rate)
    rendom_col = np.random.randint(0,len(children[0]),size=select_col)

    for i in rendom_row:
        for j in rendom_col:
            temp = np.random.normal(0, 1)
            children[i][j] = children[i][j] + temp
    return children

def survivor_selection(pop,fit_pop,offspring,fit_offspring):
    new_pop=np.concatenate([pop,offspring])
    fitness_new_pop = np.concatenate([fit_pop, fit_offspring])
    # sorted index
    sorted_index = (np.argsort(fitness_new_pop))[::-1][:len(fitness_new_pop)]
    new_pop = (new_pop[sorted_index])[:npop]
    fitness_new_pop = (fitness_new_pop[sorted_index])[:npop]
    return new_pop,fitness_new_pop



def process_results(mean, best_genomes):

    with open(r"train/StatsFile_for_EA.csv", "a") as file:
        # Get list of means

        # Clean up csv file with every run
        file.write('New Run,')

        # Loop through mean lists to add values to file
        for i in range(len(mean)):
            file.write(str(i) + ',')
            file.write(f'{mean[i]}, ')
            file.write(f'{best_genomes[i]}, ')

        file.write('\n')

if __name__ == "__main__":

    ITERATIONS = 2
    for i in range(ITERATIONS):
        all_means = []
        all_bests = []
        experiment_name = 'train/train_' + str(i + 1)
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        env.update_parameter('experiment_name', experiment_name)

        # initializes population loading old solutions or generating new ones
        if not os.path.exists(experiment_name+'/evoman_solstate'):
            print( '\nNEW EVOLUTION\n')
            pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
            fit_pop = evaluate(pop)
            best = np.argmax(fit_pop)
            mean = np.mean(fit_pop)
            all_means.append(mean)
            all_bests.append(fit_pop[best])
            std = np.std(fit_pop)
            ini_g = 0
            solutions = [pop, fit_pop]
            env.update_solutions(solutions)

        else:
            print( '\nCONTINUING EVOLUTION\n')
            env.load_state()
            pop = env.solutions[0]
            fit_pop = env.solutions[1]
            best = np.argmax(fit_pop)
            mean = np.mean(fit_pop)
            std = np.std(fit_pop)
            # finds last generation number
            file_aux  = open(experiment_name+'/gen.txt','r')
            ini_g = int(file_aux.readline())
            file_aux.close()

        # saves results for first pop
        file_aux  = open(experiment_name+'/results.txt','a')
        file_aux.write('\n\ngen best mean std')
        print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
        file_aux.close()

        # evolution
        last_sol = fit_pop[best]
        notimproved = 0

        for i in range(ini_g+1, gens):
            sorted_index = (np.argsort(fit_pop))[::-1][:len(fit_pop)] # get the indexes of sorted fitness
            offsprings,fit =parent_selection(pop[sorted_index], fit_pop[sorted_index])
            offspring = crossover(offsprings,fit)  # crossover
            mutat_offsprings=mutation(offspring)
            fit_offspring = evaluate(mutat_offsprings) # evaluation
            pop , fit_pop = survivor_selection(pop,fit_pop,offspring,fit_offspring)

            best = np.argmax(fit_pop) #best solution in generation
            fit_pop[best] = float(evaluate(np.array([pop[best]]))[0]) # repeats best eval, for stability issues
            best_sol = fit_pop[best]

            # searching new areas
            if best_sol <= last_sol:
                notimproved += 1
            else:
                last_sol = best_sol
                notimproved = 0

            if notimproved >= 15:

                file_aux  = open(experiment_name+'/results.txt','a')
                file_aux.write('\ndoomsday')
                file_aux.close()

                offsprings, fit = parent_selection(pop, fit_pop)
                offspring = crossover(offsprings, fit)  # crossover
                mutat_offsprings = mutation(offspring)
                fit_offspring = evaluate(mutat_offsprings)  # evaluation
                pop, fit_pop = survivor_selection(pop, fit_pop, offspring, fit_offspring)

                # pop, fit_pop = doomsday(pop,fit_pop)
                notimproved = 0

            best = np.argmax(fit_pop)
            std  =  np.std(fit_pop)
            mean = np.mean(fit_pop)
            all_means.append(mean)
            all_bests.append(fit_pop[best])


            # saves results
            file_aux = open(experiment_name+'/results.txt','a')
            print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
            file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
            file_aux.close()

            # saves generation number
            file_aux = open(experiment_name+'/gen.txt','w')
            file_aux.write(str(i))
            file_aux.close()

            # saves file with the best solution
            np.savetxt(experiment_name+'/best.txt',pop[best])

            # saves simulation state
            solutions = [pop, fit_pop]
            env.update_solutions(solutions)
            env.save_state()

        process_results(all_means, all_bests)

        fim = time.time() # prints total execution time for experiment
        print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')


        file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
        file.close()


        env.state_to_log() # checks environment state