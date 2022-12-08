## Neighborhood-oriented Decentralized Learning Communication in Multi-Agent System

The NOAC algorithm built on pytorch, currently it uses `pytest` as the experimental entrance.

How to run the algorithm:

`python test_main.py`

How to change algorithm hyper-parameters:

`vi tests/test_algorithms.py`

Dependencies:

`pip install -r requirements.txt`

Tensorboard logs:

`tensorboard --logdir logs --host 0.0.0.0`

## TODO

- [x] Distributed communication
- [] Fed-Avg algorithm support
- [x] Loggin and tensorboard
