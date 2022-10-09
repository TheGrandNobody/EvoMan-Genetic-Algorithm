import subprocess
import sys

for i in range(0,6):
    subprocess.Popen('python enemy_test.py ' + str(i))

