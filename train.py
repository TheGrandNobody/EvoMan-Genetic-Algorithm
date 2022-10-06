import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from controllers import specialist
import pickle
import numpy as np
import neat
from concurrent.futures import ProcessPoolExecutor


# Determines whether NEAT or the simple NN is being used
NEAT = len(sys.argv) == 1
# Holds the best genomes for each generation
best_genomes = []
# Name of the enemy
NAME = "CrashMan"
TEST = "test_generalist_9"
# Number of generations to run the simulation
GENS = 10
# Number of iterations to run each simulation
ITERATIONS = 1

# Make the module headless to run the simulation faster
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize an environment for a specialist game (single objective) with a static enemy and an ai-controlled player
env = Environment(experiment_name='logs',
              playermode="ai",
              multiplemode="yes",
              enemies=[5,6,7],
              player_controller=specialist(),
              multiplemode="yes"
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
    for genome_id, genome in genomes:
        # Create either an RNN or a simple NN for each genome
        if NEAT:
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
    with open(r"stats/%s%sStatsFile.csv" % (f"[{env.enemies}]","neat" if NEAT else "esneat"), "a") as file:
        # Get list of means
        mean = stats.get_fitness_mean()
    
        # Clean up csv file with every run
        file.write('New Run,')

        # Loop through mean lists to add values to file
        for i in range(GENS):
            file.write(str(i) + ',')
            file.write(f'{mean[i]}, ')
            file.write(f'{best_genomes[i]}, ')
        
        file.write('\n')

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
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'configs/' +  'neat-specialist.cfg')
    with ProcessPoolExecutor() as executor:
        executor.map(main, [(config, i) for i in range(ITERATIONS)])

