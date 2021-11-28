import pytest
from player import main

_default_scenario_name = None
_default_agent = None
_default_agent_path = None
_default_image_dim = 80
_default_nb_episodes = 50


def test_no_agent_name():
    with pytest.raises(ValueError):
        main(["script-name", "--scenario", "basic", "--agent_model_path", "path_agent"])


def test_no_scenario():
    with pytest.raises(ValueError):
        main(["script-name", "--agent_name", "basic", "--agent_model_path", "path_agent"])


def test_empty_agent_path():
    with pytest.raises(ValueError):
        main(["script-name", "--agent_name", "basic", "--scenario", "basic"])


def test_wrong_agent_name():
    with pytest.raises(NotImplementedError) as not_implemented:
        main(["script-name", "--agent_name", "basic2", "--scenario", "basic", "--agent_model_path", "path_agent"])
    assert str(not_implemented.value) == "There is not agent implemented for agent_name: basic2"


def test_empty_agent_name():
    with pytest.raises(NotImplementedError) as not_implemented:
        main(["script-name", "--agent_name", " ", "--scenario", "basic", "--agent_model_path", "path_agent"])
    assert str(not_implemented.value) == "There is not agent implemented for agent_name:  "
