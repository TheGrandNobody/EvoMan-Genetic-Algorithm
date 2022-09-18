import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from controllers import specialist
import numpy as np

experiment_name = 'specialist_controller'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Initializes the environment for a specialist game (single objective) with a static enemy and an ai-controlled player
env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  player_controller=specialist(),
			  	  speed="fastest",
				  enemymode="static",
				  level=2)

# saves results
file_aux  = open(experiment_name+'/results.txt','a')
print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
file_aux.close()

# saves generation number
file_aux  = open(experiment_name+'/gen.txt','w')
file_aux.write(str(i))
file_aux.close()

# saves file with the best solution
np.savetxt(experiment_name+'/best.txt',pop[best])

# saves simulation state
solutions = [pop, fit_pop]
env.update_solutions(solutions)
env.save_state()