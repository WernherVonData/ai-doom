from trainer import read_arguments


def test_correct_parameters(mocker):
    _agent_mock = mocker.path("agents.agent_basic.AgentBasic.__init__")
    read_arguments(["script-name", "--agent_name", "basic", "--scenario", "basic"])
    assert _agent_mock.call_count == 1
