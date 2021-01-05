"""
    Implement Adept DQN Agent

"""
import typing
import random
from typing import Dict, Tuple
from adept.agent import AgentModule
import torch
from adept.rewardnorm import RewardNormModule
import numpy as np

from marioRL.Adept.modules.adept_replay import AdeptMarioReplay
from marioRL.Adept.modules.adept_actor import AdeptMarioActor
from marioRL.Adept.modules.adept_learner import AdeptMarioLearner

class AdeptMarioAgent(AgentModule):
    # You will be prompted for these when training script starts
    args = {
        **AdeptMarioReplay.args,
        **AdeptMarioActor.args,
        **AdeptMarioLearner.args,
    }

    def __init__(
        self,
        reward_normalizer,
        action_space,
        spec_builder,
        exp_size,
        exp_min_size,
        exp_update_rate,
        rollout_len,
        discount,
        nb_env,
        return_scale,
        double_dqn,
    ):
        super(AdeptMarioAgent, self).__init__(
            reward_normalizer,
            action_space,
        )
        self._exp_cache = AdeptMarioReplay(
            spec_builder, exp_size, exp_min_size, rollout_len, exp_update_rate
        )
        self._actor = AdeptMarioActor(action_space, nb_env)
        self._learner = AdeptMarioLearner(
            reward_normalizer, discount, return_scale, double_dqn
        )

    @classmethod
    def from_args(
        cls, args, reward_normalizer, action_space, spec_builder, **kwargs
    ):
        return cls(
            reward_normalizer,
            action_space,
            spec_builder,
            exp_size=args.exp_size,
            exp_min_size=args.exp_min_size,
            rollout_len=args.rollout_len,
            exp_update_rate=args.exp_update_rate,
            discount=args.discount,
            nb_env=args.nb_env,
            return_scale=args.return_scale,
            double_dqn=args.double_dqn,
        )

    @property
    def exp_cache(self):
        return self._exp_cache

    @classmethod
    def _exp_spec(cls, exp_len, batch_sz, obs_space, act_space, internal_space):
        return AdeptMarioActor._exp_spec(
            exp_len, batch_sz, obs_space, act_space, internal_space
        )

    @staticmethod
    def output_space(action_space):
        return AdeptMarioActor.output_space(action_space)

    def act(self, network, obs, prev_internals):
        with torch.no_grad():
            return super().act(network, obs, prev_internals)

    def compute_action_exp(
        self, predictions, internals, obs, available_actions
    ):
        with torch.no_grad():
            return self._actor.compute_action_exp(
                predictions, internals, obs, available_actions
            )

    def learn_step(self, updater, network, next_obs, internals):
        return self._learner.learn_step(
            updater, network, self.exp_cache.read(), next_obs, internals
        )