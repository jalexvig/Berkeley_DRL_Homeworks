import logging

import numpy as np

from cost_functions import trajectory_cost_fn

logger = logging.getLogger(__name__)


class Controller():

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        raise NotImplementedError


class RandomController(Controller):

    def __init__(self, env):
        self._env = env

    def get_action(self, state):
        """ Your code should randomly sample an action uniformly from the action space """
        return self._env.action_space.sample()


class MPCcontroller(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """

    def __init__(self,
                 env,
                 dyn_model,
                 horizon=5,
                 cost_fn=None,
                 num_simulated_paths=10,
                 ):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths

    def get_action(self, state):
        """ Note: be careful to batch your simulations through the model for speed """

        # TODO(jalex): Combine starting states

        tstep_params = {
            'states': [],
            'actions': [],
            'next_states': [],
        }

        states = np.repeat(state.reshape((1, -1)), self.num_simulated_paths, axis=0)

        for t in range(self.horizon):

            actions = np.array([self.env.action_space.sample() for _ in states])

            states_next = self.dyn_model.predict(states, actions)

            tstep_params['states'].append(states)
            tstep_params['actions'].append(actions)
            tstep_params['next_states'].append(states_next)

            states = states_next

        costs = trajectory_cost_fn(self.cost_fn, **tstep_params)

        cost_argmin = np.argmin(costs)

        next_action_minimize_horizon_cost = tstep_params['actions'][0][cost_argmin]

        return next_action_minimize_horizon_cost
