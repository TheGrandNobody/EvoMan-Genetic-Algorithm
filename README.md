Evoman is a video game playing framework to be used as a testbed for optimization algorithms.
A demo can be found here:  https://www.youtube.com/watch?v=ZqaMjd1E4ZI

In order to begin training for NEAT you can call the file without any additional argument to the command line, such as:
```
python3 train.py
```
In order to begin training for ES-HyperNEAT you can simply call the same command with an additional argument, such as:
```
python3 train.py esneat
```

The training file will save the best genome as a pickle file, located in the winners folder. You must add the path to this pickle file in test.py to test it. Then, the same rules as training apply, where for NEAT you can call:
```
python3 test.py
```
and for ES-HyperNeat you can add any additional command line arguments, such as:
```
python3 train.py esn
```