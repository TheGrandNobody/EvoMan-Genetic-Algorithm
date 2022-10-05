import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from controllers import specialist
import pickle
import neat

# Whether we are training using HyperNeat or not
NEAT = len(sys.argv) == 1

# Initialize an environment for a specialist game (single objective) with a static enemy and an ai-controlled player
env = Environment(experiment_name='logs',
              playermode="ai",
              enemies=[7,8],
              player_controller=specialist(),
              speed="fastest",
              enemymode="static",
              level=2)

# Open boxplot stats file 
statsfile = open(r"Boxplot.csv", "a")
# Add enemy number & neat/esHyperneat
statsfile.write(str(env.enemies[0]) + ', ')
statsfile.write('esneat, ' if HYPERNEAT else 'neat, ')

if __name__ == "__main__":

    for i in range(0,10):
        # Initialize the NEAT config 
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'configs/' + ('esneat-specialist.cfg' if HYPERNEAT else 'neat-specialist.cfg'))
        with open('winners/CrashMan'+str(8)+'esneat-winner.pkl', "rb") as f:
            unpickler = pickle.Unpickler(f)
            genome = unpickler.load()
        # Create either an RNN or a CPPN
        if NEAT:
            rnn = neat.nn.RecurrentNetwork.create(genome, config)
    
        #statsfile.write() for formatting later
    
        a = env.play(rnn)
        print(a)
        # Write fitness value to stats file
        statsfile.write(str(a[0]) + ", ")
    
statsfile.write('\n')
statsfile.close()
