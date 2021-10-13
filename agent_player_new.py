import sys


"""
- scenario_name - Based on the scenario we will what kind of scenario play, but also what kind of image processing approach use to 
convert it to the grayscale.
- agent_type - By the agent we will know which agent to use
- agent_model_and_optimizer_path - By the path we will know where to load the agent weights and optimizer - it must be tightly connected with the agent,
unless it will not work
- image_dim Image dim is by default 80, but it should be as well matching the agent parameters.
- nb_episodes - How many episodes are we going to play. 
"""
scenario_name = ""
agent_type = ""
agent_model_and_optimizer_path = ""
image_dim = 80
nb_episodes = 50


def main(args):
    return


if __name__ == "__main__":
    main(sys.argv)
