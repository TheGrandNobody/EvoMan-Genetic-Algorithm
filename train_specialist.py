import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from controllers import specialist
import pickle
import numpy as np
import neat
from es_hyperneat import ESNetwork
from substrate import Substrate
from concurrent.futures import ProcessPoolExecutor

# Dictionary mapping names to their corresponding ids
enemies = {"WoodMan" : 3, "CrashMan" : 6, "BubbleMan" : 7}
# Holds the best genomes for each generation
best_genomes = []
# Name of the enemy
NAME = "WoodMan"
# Number of generations to run the simulation
GENS = 25
# Number of iterations to run each simulation
ITERATIONS = 5
# Whether we are training using HyperNeat or not
HYPERNEAT = len(sys.argv) > 1

# Make the module headless to run the simulation faster
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize an environment for a specialist game (single objective) with a static enemy and an ai-controlled player
env = Environment(experiment_name='logs',
              playermode="ai",
              enemies=[enemies[NAME]],
              player_controller=specialist(),
              speed="fastest",
              enemymode="static",
              level=2)

def run(config):
    # Create the population, which is the top-level object for a NEAT run.
    population = neat.Population(config)
    # Add a stdout reporter to show progress in the terminal.
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    # Run for up to 10 generations.
    return population.run(evaluate, GENS), stats

def evaluate(genomes, config):
    best = 0
    if HYPERNEAT:
        sub = Substrate(20, 5, 2, 10)
    for genome_id, genome in genomes:
        # Create either an RNN or a CPPN for each genome
        if HYPERNEAT:
            cppn = neat.nn.FeedForwardNetwork.create(genome, config)
            network = ESNetwork(sub, cppn)
            rnn = network.create_phenotype_network()
        else:
            rnn = neat.nn.RecurrentNetwork.create(genome, config)
        # Make each genome (individual) play the game
        f,p,e,t = env.play(rnn)
        # Assign a fitness value to a specific genome
        genome.fitness = 0.90*(100 - e) + 0.1*p - np.log(t)
        best = genome.fitness if genome.fitness > best else best
    best_genomes.append(best)

def process_results(winner, stats):
    # Use NEAT's Population object to obtain the statistics you want
    # Create or open a csv file called StatsFile.csv that can be written in from last position 
    file1 = open(r"stats/%s%sStatsFile.csv" % (NAME, "esneat" if HYPERNEAT else "neat"), "a")
   
    # Get list of means and stdev 
    mean = stats.get_fitness_mean()
    
    # Clean up csv file with every run
    file1.write('New Run,')

    # Loop through mean and stdev lists to add values to file
    for i in  range(GENS):
        file1.write(str(i) + ',')
        file1.write(f'{mean[i]}, ')
        file1.write(f'{best_genomes[i]}, ')
    
    file1.write('\n')
   
    # Close file
    file1.close()

def main(params) -> None:
    # Run simulations to determine a solution
    winner, stats = run(params[0])
    # Process results
    process_results(winner, stats)
    with open("winners/%s%d%s%s" % (NAME, params[1],('esneat' if HYPERNEAT else 'neat'), '-winner.pkl'), "wb") as f:
        pickle.dump(winner, f)

if __name__ == "__main__":
    # Create the folder for Assignment 1
    if not os.path.exists('logs'):
        os.makedirs('logs')
    # Initialize the NEAT config 
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'configs/' + ('esneat-specialist.cfg' if HYPERNEAT else 'neat-specialist.cfg'))
    with ProcessPoolExecutor() as executor:
        executor.map(main, [(config, i) for i in range(ITERATIONS)])

