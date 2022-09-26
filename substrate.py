"""
The substrate that is optimized by the CPPN.
"""

class Substrate(object):
    """
    Represents a substrate: Input coordinates, output coordinates, hidden coordinates and a resolution defaulting to 10.0.
    Input and output coordinates are automatically calculated based on the number of inputs and outputs.
    There is one hidden layer by default.
    """

    def __init__(self, num_inputs, num_outputs, hidden_coordinates=(), res=10.0):
        self.input_coordinates = [(float(i), -1.0) for i in range(-20//2, 20//2)]
        self.output_coordinates = [(float(i), 1.0) for i in range(-2, 3)]
        self.hidden_coordinates = [[(float(i), 0.0) for i in range(-10//2, 10//2)]]
        self.res = res
