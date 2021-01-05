import torch
import numpy as np

from adept.learner.base.learner_module import LearnerModule
from adept.learner.base.dm_return_scale import DeepMindReturnScaler


class AdeptMarioLearner(LearnerModule):
    args = {"discount": 0.99, "return_scale": False, "double_dqn": True}

    def __init__(self, reward_normalizer, discount, return_scale, double_dqn):
        self.reward_normalizer = reward_normalizer
        self.discount = discount
        self.return_scale = return_scale
        self.double_dqn = double_dqn
        if return_scale:
            self.dm_scaler = DeepMindReturnScaler(100.0 ** -3)

    @classmethod
    def from_args(cls, args, reward_normalizer):
        return cls(
            reward_normalizer, args.discount, args.return_scale, args.double_dqn
        )

    def _get_q_vals_from_pred(self, preds):
        return preds

    def _action_from_q_vals(self, q_vals):
        return q_vals.argmax(dim=-1, keepdim=True)

    def _get_action_values(self, q_vals, action, batch_size=0):
        return q_vals.gather(1, action)

    def learn_step(
        self, updater, network, experiences, next_obs, current_internals
    ):
        # normalize rewards
        rewards = self.reward_normalizer(torch.stack(experiences.rewards))

        # first internals
        internals = {
            k: v.unbind(0) for k, v in experiences.internals[0].items()
        }

        # iterate observations and internals to generate values
        batch_values = []
        # keep a copy of terminals on the cpu it's faster
        rollout_terminals = torch.stack(experiences.terminals).cpu().numpy()
        batch_size = rollout_terminals.shape[1]

        for obs, action, terminals in zip(
            experiences.observations, experiences.actions, rollout_terminals
        ):
            predictions, internals, _ = network(obs, internals)
            # TODO: HACK for QR and DDQN , where can this class get action keys?
            self.action_keys = list(
                filter(lambda x: x != "value", predictions.keys())
            )
            q_vals = self._get_q_vals_from_pred(predictions)

            # select from values based on a samples action
            q_val_action = [
                self._get_action_values(q_vals[k], action[k], batch_size)
                for k in self.action_keys
            ]
            batch_values.append(torch.stack(q_val_action, dim=1))

            # where returns a single element tuple with the indices
            terminal_inds = np.where(terminals)[0]
            for i in terminal_inds:
                for k, v in network.new_internals(
                    next(network.parameters()).device
                ).items():
                    internals[k][i] = v

            # estimate value for next state
            last_values = self.compute_estimated_values(
                network, network, next_obs, internals
            )

            # compute nstep return and advantage over batch
            batch_values = torch.stack(batch_values)
            value_targets = self.compute_returns(
                last_values, rewards, experiences.terminals
            )

            # batched loss
            value_loss = self.loss_fn(batch_values, value_targets)

            losses = {"value_loss": value_loss.mean()}
            metrics = {}
            return losses, metrics

    def loss_fn(self, batch_values, value_targets):
        return 0.5 * (value_targets - batch_values).pow(2)

    def compute_estimated_values(
        self, network, target_network, next_obs, internals
    ):
        # estimate value of next state
        with torch.no_grad():
            results, _, _ = target_network(next_obs, internals)
            target_q = self._get_q_vals_from_pred(results)
            batch_size = target_q[self.action_keys[0]].shape[0]

            # if double dqn estimate, get target val for fcurrent estimate action
            if self.double_dqn:
                current_results, _, _ = network(next_obs, internals)
                current_q = self._get_q_vals_from_pred(current_results)
                last_actions = [
                    self._action_from_q_vals(current_q[k])
                    for k in self.action_keys
                ]
                last_values = []
                for k, a in zip(self.action_keys, last_actions):
                    last_values.append(
                        self._get_action_values(target_q[k], a, batch_size)
                    )
                last_values = torch.stack(last_values, dim=1)
            else:
                # TODO this should be a function so it can be overridden
                last_values = torch.stack(
                    [
                        torch.max(target_q[k], 1)[0].data
                        for k in self.action_keys
                    ],
                    dim=1,
                )

        # nb_env, nb_action, value_size
        return last_values

    def compute_returns(self, estimated_value, rewards, terminals):
        next_value = estimated_value
        # First step of nstep reward target is estimate d value of t+1
        target_return = estimated_value
        nstep_target_returns = []
        for i in reversed(range(len(rewards))):
            # unsqueeze over action dim so it isn't broadcasted
            reward = rewards[i].unsqueeze(-1).unsqueeze(-1)
            terminal_mask = (
                1.0 - terminals[i].unsqueeze(-1).unsqueeze(-1).float()
            )

            # Nstep return is always calculated for the critic's target
            if self.return_scale:
                target_return = self.dm_scaler.calc_scale(
                    reward + self.discount * target_return * terminal_mask
                )
            else:
                target_return = (
                    reward + self.discount * target_return * terminal_mask
                )
            nstep_target_returns.append(target_return)
        # reverse lists
        nstep_target_returns = torch.stack(
            list(reversed(nstep_target_returns))
        ).data
        return nstep_target_returns