import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from controllers import specialist
import pickle
import neat

# Whether we are training using HyperNeat or not
HYPERNEAT = len(sys.argv) > 1

# Initialize an environment for a specialist game (single objective) with a static enemy and an ai-controlled player
env = Environment(experiment_name='logs',
              playermode="ai",
              multiplemode="yes",
              enemies=[1,2,3,4,5,6,7,8],
              player_controller=specialist(),
              speed="fastest",
              multiplemode="no"
              enemymode="static",
              level=2)

# Open boxplot stats file 
statsfile = open(r"test.csv", "a")
statsfile.write('Test7' + ', ')

if __name__ == "__main__":

    for i in range(0,3):
        statsfile.write(str(i) + ', ')
        # Initialize the NEAT config 
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'configs/' + ('esneat-specialist.cfg' if HYPERNEAT else 'neat-specialist.cfg'))
        with open('winners/test_generalist_7'+str(i)+'neat-winner.pkl', "rb") as f:
            unpickler = pickle.Unpickler(f)
            genome = unpickler.load()
        # Create either an RNN or a CPPN
        rnn = neat.nn.RecurrentNetwork.create(genome, config)
    
        #statsfile.write() for formatting later
    
        a = env.play(rnn)
        print(a)
        # Write fitness value to stats file
        statsfile.write(str(a[0]) + ", ")
    
statsfile.write('\n')
statsfile.close()
