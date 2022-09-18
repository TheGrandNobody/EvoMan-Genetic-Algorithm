import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from controllers import specialist
import numpy as np
import neat

def run(env, config):
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
        # Make each genome (individual) play the game
        # Save their fitness 
        pass

def process_results(winner, config):
    # Use NEAT's Population object to obtain the statistics you want
    # Check out opitmization_specialist_demo.py to see what part of the results-writing code you can take
    pass

if __name__ == "__main__":
    # Create the folder for Assignment 1
    experiment_name = 'A1_specialist'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Initialize the NEAT config 
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'neat-specialist')

    # Initialize the environment for a specialist game (single objective) with a static enemy and an ai-controlled player
    env = Environment(experiment_name=experiment_name,
              playermode="ai",
              player_controller=specialist(),
              speed="fastest",
              enemymode="static",
              level=2)

    # Make the environment headless to run faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    # Run simulations to determine a solution
    winner = run(env, config)
    # Process results
    process_results(winner, config)