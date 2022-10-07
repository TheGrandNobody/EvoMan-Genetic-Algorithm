import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from controllers import specialist
import pickle
import numpy as np
import neat
from concurrent.futures import ProcessPoolExecutor
from extra.substrate import Substrate
#from extra.es_hyperneat import ESNetwork
from extra.hyperneat import create_phenotype_network

# Determines whether NEAT or the simple NN is being used (Changed to 2 to allow for running multiple tests)
# The second argv gives a reference for the enemy-list
NEAT = len(sys.argv) == 2
# Holds the best genomes for each generation
best_genomes = []
# Data set for enemy tests. First column is the test number, the second column is a list of enemies
enemy_list =    [[4,[1,3,2]],
                [5,[3,6,7]],
                [6,[1,2,8]],
                [7,[1,7,8]],
                [9,[5,6,7]],
                [11,[7,8]]]
# Number of generations to run the simulation
GENS = 10
# Number of iterations to run each simulation
ITERATIONS = 3


# Make the module headless to run the simulation faster
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Name of the enemy
TEST = "test_generalist_"+str(enemy_list[int(sys.argv[1])][0])

# Initialize an environment for a specialist game (single objective) with a static enemy and an ai-controlled player
env = Environment(experiment_name='logs',
              playermode="ai",
              enemies=enemy_list[int(sys.argv[1])][1],
              player_controller=specialist(),
              multiplemode="yes",
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
    if not NEAT:
        sub = Substrate(20, 5)
    for genome_id, genome in genomes:
        # Create either an RNN or a simple NN for each genome
        if NEAT:
            nn = neat.nn.FeedForwardNetwork.create(genome, config)
        # Make each genome (individual) play the game
        f,p,e,t = env.play(nn)
        # Assign a fitness value to a specific genome
        genome.fitness = 0.90*(100 - e) + 0.1*p - np.log(t)
        best = genome.fitness if genome.fitness > best else best
    best_genomes.append(best)

def main(params) -> None:
    # Run simulations to determine a solution
    winner, stats = run(params[0])
    # Process results
    # process_results(winner, stats)
    with open("winners/%s%d%s%s" % (TEST, params[1], 'neat', '-winner.pkl'), "wb") as f:
        pickle.dump(winner, f)

if __name__ == "__main__":
    # Create the folder for Assignment 2
    if not os.path.exists('logs'):
        os.makedirs('logs')
    # Initialize the NEAT config 
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'configs/' +  'neat-generalist.cfg')
    with ProcessPoolExecutor() as executor:
        executor.map(main, [(config, i) for i in range(ITERATIONS)])

