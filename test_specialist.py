import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from controllers import specialist
import pickle
import neat
from es_hyperneat import ESNetwork
from substrate import Substrate

# Whether we are training using HyperNeat or not
HYPERNEAT = len(sys.argv) > 1

# Initialize an environment for a specialist game (single objective) with a static enemy and an ai-controlled player
env = Environment(experiment_name='logs',
              playermode="ai",
              enemies=[7],
              player_controller=specialist(),
              speed="fastest",
              enemymode="static",
              level=2)

if __name__ == "__main__":
    # Initialize the NEAT config 
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'configs/' + ('esneat-specialist.cfg' if HYPERNEAT else 'neat-specialist.cfg'))
    with open(('esneat' if HYPERNEAT else 'neat') + '-winner.pkl', "rb") as f:
        unpickler = pickle.Unpickler(f)
        genome = unpickler.load()
    # Create either an RNN or a CPPN
    if HYPERNEAT:
        sub = Substrate(20, 5)
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        network = ESNetwork(sub, cppn)
        rnn = network.create_phenotype_network()
    else:
        rnn = neat.nn.RecurrentNetwork.create(genome, config)
    # Make the genome (individual) play the game
    print(env.play(rnn))