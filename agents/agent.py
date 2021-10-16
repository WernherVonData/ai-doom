class Agent:

    def read_state(self, state, image_dim):
        raise NotImplementedError("Method {} must be implemented in the child class".format("read_state"))
        yield

    def make_action(self, ai, state_data):
        raise NotImplementedError("Method {} must be implemented in the child class".format("make_action"))
        yield

    def calculate_reward(self, game_reward):
        raise NotImplementedError("Method {} must be implemented in the child class".format("calculate_reward"))
        yield

    def generate_history_record(self, action, game_finished):
        raise NotImplementedError("Method {} must be implemented in the child class".format("generate_history_record"))
        yield

    def eligibility_trace(self, cnn, batch, gamma):
        raise NotImplementedError("Method {} must be implemented in the child class".format("eligibility_trace"))
        yield
