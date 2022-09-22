import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from controllers import specialist
import numpy as np
import neat

# Create the folder for Assignment 1
if not os.path.exists('A1_specialist'):
    os.makedirs('A1_specialist')

# Initialize an environment for a specialist game (single objective) with a static enemy and an ai-controlled player
env = Environment(experiment_name='A1_specialist',
              playermode="ai",
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
    # Saves the state of the simulation every 5 generations (optional)
    population.add_reporter(neat.Checkpointer(5))

    # Run for up to 10 generations.
    return population.run(evaluate, 10)

def evaluate(genomes, config):
    for genome_id, genome in genomes:
        # Create a neural network for each genome
        rnn = neat.nn.RecurrentNetwork.create(genome, config)
        # Make each genome (individual) play the game
        f,p,e,t = env.play(rnn)
        # Assign a fitness value to a specific genome
        genome.fitness = f
        # genome.fitness = 0.9*(100 - e) + 0.1*p - np.log(t)

def process_results(winner, config):
    # Use NEAT's Population object to obtain the statistics you want
    # Check out opitmization_specialist_demo.py to see what part of the results-writing code you can take
    pass

if __name__ == "__main__":
    # Initialize the NEAT config 
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, os.path.join(os.path.dirname(__file__), 'neat-specialist'))

    # Make the environment headless to run faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    # Run simulations to determine a solution
    winner = run(config)
    # Process results
    process_results(winner, config)