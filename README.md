# ai-doom

The training agent wrapper for Vizdoom environment. 

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Running](#running)
4. [Licensing, Authors and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

For the beginning install the [Vizdoom](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md) - 
follow the instructions for your platform. I'm bound to Windows for some time - I was using the built binary prepared by the author (available on that project repo).

Before installing the requirement install the [pytorch](https://pytorch.org/get-started/locally/) on your own (you can get there the best configuration on your platform/requirements) - installing directly using the command below will install the default versions for your pip (which you might not want to have).

Then just run:
```
pip install -r requirements.txt
```

## Project Motivation <a name="motivation"></a>

During work on the AI course I found it's content outdated when it comes to the used libraries (the core code is in 40% from there, but it's wrapped in custom implementations).

## Running <a name="running"></a>

To run the most basic example just call:

```
python trainer.py --agent_name basic --scenario basic
```

This will call agent training using the basic agent along with the most basic scenario from [Vizdoom](https://github.com/mwydmuch/ViZDoom/tree/master/scenarios).

For script `trainer.py` the available options are:
* --agent_name - the name of the agent, by default the available are: 'basic' and 'linear'.
* --agent_model_path - the path to the serialized agent and optimizer.
* --scenario - the scenario to train - available are (.cfg files are Vizdoom scenarios names): 'basic' (basic.cfg), 'rocket' (rocket_basic.cfg) and 'corridor' (deadly_corridor.cfg).
* --starting_epoch - number of the epoch that we are going to start.
* --nb_epochs - for how many epochs we are going to train the agent.
* --n_step -  how many steps the agent will perform in the environment before saving them as a memory record.
* --n_steps - how many records will be saved to the memory before training will happen.
* --memory_capacity - how many entries to memory we will have.
* --memory_path - path to the memory file, is set to None the learning will start with empty memory.
* --image_dim - the dimension in pixels of read image (we are reading them as square)

To play trained agent we can use script `player.py` for which the options are:
* --scenario - same as for 'trainer.py'
* --agent_name - same as for 'trainer.py'
* --agent_model_path - same as for 'trainer.py'
* --image_dim - same as for 'trainer.py'
* --nb_episodes - how many episodes (full gane ended with reaching goal/losing) we play

## Licensing, Authors and Acknowledgements <a name="licensing"></a>

Project is released under MIT License I would like to get information whether the project was useful for you. In case of issues/bug don't hestitate to use the issue tracker for this repo on github.