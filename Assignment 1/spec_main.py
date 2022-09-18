import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from controllers import specialist
import numpy as np

experiment_name = 'A1_specialist'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Initializes the environment for a specialist game (single objective) with a static enemy and an ai-controlled player
env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  player_controller=specialist(),
			  	speed="fastest",
				  enemymode="static",
				  level=2)

# 
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"