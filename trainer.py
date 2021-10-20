from agents import agent_basic
import agent_trainer


def main():
    agent = agent_basic.AgentBasic(scenario_name="basic", agent_identifier="agent_basic", image_dim=80)
    trainer = agent_trainer.AgentTrainer(agent_to_train=agent, memory_capacity=1000)
    trainer.train(nb_epochs=50, n_step=10, n_steps=200)
    return


if __name__ == "__main__":
    main()
