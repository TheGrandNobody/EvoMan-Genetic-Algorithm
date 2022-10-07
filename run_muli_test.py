import subprocess
import sys

# Data set for enemy tests
enemy_list =    [[4,[1,3,2]],
                [5,[3,6,7]],
                [6,[1,2,8]],
                [7,[1,7,8]],
                [9,[5,6,7]],
                [11,[7,8]]]

for i in range(0,6):
    subprocess.Popen('python enemy_test.py ' + str(i))

