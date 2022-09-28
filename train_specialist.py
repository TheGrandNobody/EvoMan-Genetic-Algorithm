import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from controllers import specialist
import pickle
import numpy as np
import neat
from es_hyperneat import ESNetwork
from substrate import Substrate

enemies = {"WoodMan" : 3, "CrashMan" : 6, "BubbleMan" : 7}
# Name of the enemy
NAME = "WoodMan"
# Number of generations to run the simulation
GENS = 25
# Holds the best genomes for each generation
best_genomes = []

# Whether we are training using HyperNeat or not
HYPERNEAT = len(sys.argv) > 1

# Make the module headless to run the simulation faster
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize an environment for a specialist game (single objective) with a static enemy and an ai-controlled player
env = Environment(experiment_name='A1_specialist',
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
    # Check out opitmization_specialist_demo.py to see what part of the results-writing code you can take
    # Create or open a csv file called StatsFile.csv that can be written in from last position 
    file1 = open(r"%sStatsFile.csv" % (NAME), "a")
   
    # Get list of means and stdev 
    mean = stats.get_fitness_mean()
    
    # Clean up csv file with every run
    file1.write('New Run,')

    # Loop through mean and stdev lists to add values to file
    for i in  range(GENS):
        file1.write(str(i) + ',')
        file1.write(f'{mean[i]}, ')
        file1.write(f'{best_genomes[i]}, ')
   
    # Check inbuilt fitness mean and max saver from NEAT, saves to separate SaveGenomeFitness.csv
    # TODO rewrite NEAT's function so that it does not rewrite file on each run, and to clean up data
    # Close file
    file1.close()

if __name__ == "__main__":
    # Create the folder for Assignment 1
    if not os.path.exists('A1_specialist'):
        os.makedirs('A1_specialist')
    # Initialize the NEAT config 
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'configs/' + ('esneat-specialist.cfg' if HYPERNEAT else 'neat-specialist.cfg'))

    for i in range(5):
        # Run simulations to determine a solution
        winner, stats = run(config)
        # Process results
        process_results(winner, stats)
        with open((NAME + str(i) + 'esneat' if HYPERNEAT else 'neat') + '-winner.pkl', "wb") as f:
            pickle.dump(winner, f)
