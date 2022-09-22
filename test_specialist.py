import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from controllers import specialist
import pickle
import neat

# Initialize an environment for a specialist game (single objective) with a static enemy and an ai-controlled player
env = Environment(experiment_name='A1_specialist',
              playermode="ai",
              enemies=[7],
              player_controller=specialist(),
              speed="fastest",
              enemymode="static",
              level=2)

if __name__ == "__main__":
    # Initialize the NEAT config 
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'neat-specialist.cfg')
    with open("winner.pkl", "rb") as f:
        unpickler = pickle.Unpickler(f)
        genome = unpickler.load()
    rnn = neat.nn.RecurrentNetwork.create(genome, config)
    # Make each genome (individual) play the game
    print(env.play(rnn))
    
    