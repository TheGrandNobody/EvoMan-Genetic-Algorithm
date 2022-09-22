import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from controllers import specialist
from ahyperneat.adaptive_linear_net import AdaptiveLinearNet
from ahyperneat.activations import tanh_activation
import pickle
import numpy as np
import neat

hyperneat = True

# Create the folder for Assignment 1
if not os.path.exists('A1_specialist'):
    os.makedirs('A1_specialist')

# Make the module headless to run the simulation faster
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize an environment for a specialist game (single objective) with a static enemy and an ai-controlled player
env = Environment(experiment_name='A1_specialist',
              playermode="ai",
              enemies=[7],
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
    return population.run(evaluate, 10), stats

def evaluate(genomes, config):
    for genome_id, genome in genomes:
        # Create a neural network for each genome
        if hyperneat:
            rnn = AdaptiveLinearNet.create(
                genome,
                config,
                input_coords=[[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, -1.0]],
                output_coords=[[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
                weight_threshold=0.4,
                batch_size=1,
                activation=tanh_activation,
                output_activation=tanh_activation,
                device="cpu",
            )
        else:
            rnn = neat.nn.RecurrentNetwork.create(genome, config)
        # Make each genome (individual) play the game
        f,p,e,t = env.play(rnn)
        # Assign a fitness value to a specific genome
        genome.fitness = 0.90*(100 - e) + 0.1*p - np.log(t)

def process_results(winner, config):
    # Use NEAT's Population object to obtain the statistics you want
    # Check out opitmization_specialist_demo.py to see what part of the results-writing code you can take
    pass

  

if __name__ == "__main__":
    # Initialize the NEAT config 
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'neat-specialist.cfg')
    # Run simulations to determine a solution
    winner, stats = run(config)
    # Process results
    process_results(winner, config)
    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
    


