import sys
sys.path.insert(0, 'evoman')
sys.path.insert(1, 'extra')
from environment import Environment
from controllers import specialist
import pickle
import neat
from extra.es_hyperneat import ESNetwork
from extra.hyperneat import create_phenotype_network
from extra.substrate import Substrate

# Whether we are training using HyperNeat or not
NEAT = len(sys.argv) == 1

# Give list of enemies it was trained on (with spacec inbetween, no comma's)
ENEMIES_GENERAL = "1 2 3 4 5 6 7 8"

# Initialize an environment for a specialist game (single objective) with a static enemy and an ai-controlled player
env = Environment(experiment_name='logs',
              playermode="ai",
              multiplemode="yes",
              enemies=[1,2,3,4,5,6,7,8],
              player_controller=specialist(),
              speed="fastest",
              enemymode="static",
              level=2)

# Open boxplot stats file 
statsfile = open(r"test.csv", "a")
statsfile.write("test_17,"+ "neat," if NEAT else "esneat,")

if __name__ == "__main__":

    for i in range(0,3):
        statsfile.write(str(i) + ',')
        # Initialize the NEAT config 
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'configs/' + ('neat-generalist.cfg' if NEAT else 'esneat-generalist.cfg'))
        with open('winners/test_generalist_17'+str(i)+'neat-winner.pkl', "rb") as f:
            unpickler = pickle.Unpickler(f)
            genome = unpickler.load()
        # Create either an Feedforward Network or a CPPN
        if NEAT:
            rnn = neat.nn.FeedForwardNetwork.create(genome, config)
        else:
            sub = Substrate(20, 5)
            cppn = neat.nn.FeedForwardNetwork.create(genome, config)
            network = ESNetwork(sub, cppn)
            nn = network.create_phenotype_network()
    
        a = env.play(rnn)
        print(a)
        # Write fitness value to stats file
        statsfile.write(str(a[2] - a[1]) + ",")
    
statsfile.write('\n')
statsfile.close()
