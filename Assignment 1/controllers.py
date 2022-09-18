from typing import List
from controller import Controller
import numpy as np
import neat

class specialist(Controller):
    """ Implements a controller for a "specialist" player.

    Args:
        Controller (_type_): The EvoMan controller class
    """

    def control(self, x, inputs: np.ndarray, controller: np.ndarray) -> List[int]:
        """ Decides which actions the Specialist player should perform.

        Args:
            inputs (np.ndarray): 20 measurements related to the ongoing game, obtained by a Sensor object.
            controller (np.ndarray): The weights of the Specialist player's neural network.

        Returns:
            List[int]: A list containing integers to be interpreted as boolean values for each possible action.
        """
        # INSERT CODE IMPLEMENTING THE NEURAL NETWORK COMPUTATION OF THE OUTPUT USING THE SENSOR INPUTS
        # CODE SHOULD END WITH A VARIABLE CALLED "output" CONTAINING EACH ACTIVATION VALUE FOR EACH ACTION

        return [1 if i > 0.5 else 0 for i in output]

class enemy(Controller):
    """ Implements the enemy controller as seen in demo_controller.py

    Args:
        Controller (_type_): The EvoMan controller class
    """

    def __init__(self, _n_hidden) -> None:
        # Number of hidden neurons
        self.n_hidden = [_n_hidden]

    def sigmoid_activation(self, x: int) -> int:
        """ Computes a sigmoid activation function of an NN's given weighted sum of inputs.

        Args:
            x (int): The specified weighted sum of inputs obtained from a neural network layer.

        Returns:
            int: A non-linear function of the weighted sum of inputs (lying between 0 and 1)
        """
        return 1./(1.+np.exp(-x))

    def control(self, inputs: np.ndarray, controller: np.ndarray) -> List[int]:
        """_summary_

        Args:
            inputs (np.ndarray): 20 measurements related to the ongoing game, obtained by a Sensor object.
            controller (np.ndarray): The weights of the enemy's neural network.

        Returns:
            List[int]: A list containing integers to be interpreted as boolean values for each possible action.
        """
        # Normalises the input using min-max scaling
        inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

        if self.n_hidden[0] > 0:
            # Preparing the weights and biases from the controller of layer 1

            # Biases for the n hidden neurons
            bias1 = controller[:self.n_hidden[0]].reshape(1, self.n_hidden[0])
            # Weights for the connections from the inputs to the hidden nodes
            weights1_slice = len(inputs)*self.n_hidden[0] + self.n_hidden[0]
            weights1 = controller[self.n_hidden[0]:weights1_slice].reshape(
                (len(inputs), self.n_hidden[0]))

            # Outputs activation first layer.
            output1 = self.sigmoid_activation(inputs.dot(weights1) + bias1)

            # Preparing the weights and biases from the controller of layer 2
            bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1, 5)
            weights2 = controller[weights1_slice +
                                  5:].reshape((self.n_hidden[0], 5))

            # Outputting activated second layer. Each entry in the output is an action
            output = self.sigmoid_activation(output1.dot(weights2) + bias2)[0]
        else:
            bias = controller[:5].reshape(1, 5)
            weights = controller[5:].reshape((len(inputs), 5))

            output = self.sigmoid_activation(inputs.dot(weights) + bias)[0]

        # takes decisions about sprite actions
        if output[0] > 0.5:
            attack1 = 1
        else:
            attack1 = 0

        if output[1] > 0.5:
            attack2 = 1
        else:
            attack2 = 0

        if output[2] > 0.5:
            attack3 = 1
        else:
            attack3 = 0

        if output[3] > 0.5:
            attack4 = 1
        else:
            attack4 = 0

        return [attack1, attack2, attack3, attack4]
