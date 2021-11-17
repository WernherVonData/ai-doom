import sys

from agents import agent_basic
import agent_trainer


def read_arguments(args):
    _agent_name_arg = "--agent_name"
    _agent_model_path_arg = "--agent_model_path"
    _scenario_name_arg = "--scenario"
    _starting_epoch_arg = "--starting_epoch"
    _nb_epochs_arg = "--nb_epochs"
    _image_dim_arg = "--image_dim"
    _memory_capacity_arg = "--memory_capacity"
    _memory_path_arg = "--memory_path"
    _n_step_arg = "--n_step"
    _n_steps_arg = "--n_steps"

    agent_name = None
    agent_path = None
    scenario_name = None
    starting_epoch = 1
    nb_epochs = 50
    image_dim = 80
    memory_capacity = 1000
    memory_path = None
    n_step = 10
    n_steps = 100

    for i in range(1, len(args)):
        if args[i] in _agent_name_arg:
            agent_name = args[i + 1]
            i += 1
            continue
        if args[i] in _agent_model_path_arg:
            agent_path = args[i + 1]
            i += 1
            continue
        if args[i] in _scenario_name_arg:
            scenario_name = args[i + 1]
            i += 1
            continue
        if args[i] in _starting_epoch_arg:
            starting_epoch = args[i + 1]
            i += 1
            continue
        if args[i] in _nb_epochs_arg:
            nb_epochs = args[i + 1]
            i += 1
            continue
        if args[i] in _image_dim_arg:
            image_dim = args[i + 1]
            i += 1
            continue
        if args[i] in _memory_capacity_arg:
            memory_capacity = args[i + 1]
            i += 1
            continue
        if args[i] in _memory_path_arg:
            memory_path = args[i + 1]
            i += 1
            continue
        if args[i] in _n_step_arg:
            n_step = args[i + 1]
            i += 1
            continue
        if args[i] in _n_steps_arg:
            n_steps = args[i + 1]
            i += 1
            continue
    return agent_name, agent_path, scenario_name, starting_epoch, nb_epochs, image_dim, memory_capacity, memory_path, n_step, n_steps


def main(args):
    agent_name, agent_path, scenario_name, starting_epoch, nb_epochs, image_dim, memory_capacity, memory_path, n_step, n_steps = read_arguments(
        args=args)
    if agent_name is None or scenario_name is None:
        raise ValueError("--agent_name and --scenario must be set.")
    agent = None
    if agent_name is "basic":
        agent = agent_basic.AgentBasic(scenario_name=scenario_name, agent_identifier=agent_name, image_dim=80)
    if agent is None:
        raise NotImplementedError("There is not agent implemented for agent_name: {}".format(agent_name))

    if agent_path is not None:
        print("> Loading agent from file for training continuation")
        agent.load_agent_optimizer(model_path=agent_path)
    trainer = agent_trainer.AgentTrainer(agent_to_train=agent, memory_capacity=memory_capacity)
    trainer.train(memory_path=memory_path, n_step=n_step, n_steps=n_steps, nb_epochs=nb_epochs,
                  starting_epoch=starting_epoch)
    return


if __name__ == "__main__":
    main(sys.argv)
