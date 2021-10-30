from trainer import main


def test_correct_parameters(mocker):
    _trainer_mock = mocker.patch("trainer.agent_trainer.AgentTrainer.train")
    main(["script-name", "--agent_name", "basic", "--scenario", "basic"])
    print("DDDUUUPPPAAA")
    print(_trainer_mock.call_args_list)
    print(_trainer_mock.call_args)
    _trainer_mock.assert_called_once()
    assert _trainer_mock.call_count == 0

