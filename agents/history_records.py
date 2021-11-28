from collections import namedtuple

BasicStep = namedtuple('BasicStep', ['state', 'action', 'reward', 'done'])
LinearStep = namedtuple('LinearStep', ['state', 'linear', 'action', 'reward', 'done'])