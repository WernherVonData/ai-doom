import pytest
from trainer import main

_default_agent_name = None
_default_agent_path = None
_default_scenario_name = None
_default_starting_epoch = 1
_default_nb_epochs = 50
_default_image_dim = 80
_default_memory_capacity = 1000
_default_memory_path = None
_default_n_step = 10
_default_n_steps = 100


def test_correct_default_parameters(mocker):
    _trainer_mock = mocker.patch("trainer.agent_trainer.AgentTrainer.train")
    main(["script-name", "--agent_name", "basic", "--scenario", "basic"])
    _trainer_mock.assert_called_once()
    _trainer_mock.assert_called_with(memory_path=_default_memory_path, n_step=_default_n_step, n_steps=_default_n_steps,
                                     nb_epochs=_default_nb_epochs,
                                     starting_epoch=_default_starting_epoch)


def test_passed_with_agent_path(mocker):
    _load_agent_optimizer_mock = mocker.patch("trainer.agent_basic.AgentBasic.load_agent_optimizer")
    _trainer_mock = mocker.patch("trainer.agent_trainer.AgentTrainer.train")
    main(["script-name", "--agent_name", "basic", "--scenario", "basic", "--agent_model_path", "path_agent"])
    _load_agent_optimizer_mock.assert_called_once()
    _trainer_mock.assert_called_once()
    _trainer_mock.assert_called_with(memory_path=_default_memory_path, n_step=_default_n_step, n_steps=_default_n_steps,
                                     nb_epochs=_default_nb_epochs,
                                     starting_epoch=_default_starting_epoch)


def test_empty_script_name():
    with pytest.raises(ValueError):
        main(["script-name", "--scenario", "basic"])


def test_empty_agent_name():
    with pytest.raises(ValueError):
        main(["script-name", "--agent_name", "basic"])
