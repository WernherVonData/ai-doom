import sys

from agents import agent_basic
import agent_trainer


def read_arguments(args):
    _agent_name_arg = "--agent_name"
    _scenario_name_arg = "--scenario"
    _agent_model_path_arg = "--agent_model_path"
    _nb_epochs_arg = "--nb_epochs"
    _image_dim_arg = "--image_dim"
    _memory_capacity_arg = "--memory_capacity"
    _memory_path_arg = "--memory_path"
    _n_step_arg = "--n_step"
    _n_steps_arg = "--n_steps"

    agent_name = None
    agent_path = None
    scenario_name = None
    memory_path = None
    nb_epochs = 50
    n_step = 10
    n_steps = 100
    memory_capacity = 1000
    image_dim = 80


def main(args):
    agent = agent_basic.AgentBasic(scenario_name="basic", agent_identifier="agent_basic", image_dim=80)
    trainer = agent_trainer.AgentTrainer(agent_to_train=agent, memory_capacity=1000)
    trainer.train(nb_epochs=50, n_step=10, n_steps=200)
    return


if __name__ == "__main__":
    main(sys.argv)
