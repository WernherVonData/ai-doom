import sys

import utils

"""
- scenario_name - Based on the scenario we will what kind of scenario play, but also what kind of image processing approach use to 
convert it to the grayscale.
    - basic
    - rocket_basic
    - corridor
- agent_type - By the agent we will know which agent to use
- agent_model_and_optimizer_path - By the path we will know where to load the agent weights and optimizer - it must be tightly connected with the agent,
unless it will not work
- image_dim Image dim is by default 80, but it should be as well matching the agent parameters.
- nb_episodes - How many episodes are we going to play. 
"""


def read_from_arguments(args):
    _scenario_name_arg = "--scenario"
    _agent_name_arg = "--agent_name"
    _agent_model_path_arg = "--agent_model_path"
    _image_dim_arg = "--image_dim"
    _nb_episodes_arg = "--nb_episodes"

    scenario_name = None
    agent = None
    agent_path = None
    image_dim = 80
    nb_episodes = 50
    for i in range(1, len(args)):
        if args[i] in _scenario_name_arg:
            scenario_name = args[i+1]
            i += 1
            continue
        if args[i] in _agent_name_arg:
            agent = args[i+1]
            i += 1
            continue
        if args[i] in _agent_model_path_arg:
            agent_path = args[i + 1]
            i += 1
            continue
        if args[i] in _image_dim_arg:
            image_dim = args[i+1]
            i += 1
            continue
        if args[i] in _nb_episodes_arg:
            nb_episodes = args[i+1]
            i += 1
            continue
    return scenario_name, agent, agent_path, image_dim, nb_episodes


def main(args):
    scenario_name, agent, agent_path, image_dim, nb_episodes = read_from_arguments(args)
    if scenario_name or agent or agent_path is None:
        raise ValueError("Scenario, agent and path to serialized agent MUST be specified")

    print("=>device used: {}".format(utils.DEVICE_NAME))
    scenario, nb_available_buttons = utils.get_path_and_number_of_actions_to_scenario(scenario_name=scenario_name)
    actions = []
    for i in range(0, nb_available_buttons):
        actions.append([True if action_index == i else False for action_index in range(0, nb_available_buttons)])
    number_actions = len(actions)

    return


if __name__ == "__main__":
    main(sys.argv)
