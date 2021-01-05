from collections import OrderedDict

import torch

from adept.actor.base.actor_module import ActorModule


class AdeptMarioActor(ActorModule):
    args = {}

    def __init__(self, action_space, nb_env):
        super().__init__(action_space)
        # use ape-x epsilon
        self.epsilon = 0.4 ** (
            1
            + (
                (
                    torch.arange(nb_env, dtype=torch.float, requires_grad=False)
                    / (nb_env - 1)
                )
                * 7
            )
        )

    @classmethod
    def from_args(cls, args, action_space):
        return cls(action_space, args.nb_env)

    @staticmethod
    def output_space(action_space):
        head_dict = {**action_space}
        return head_dict

    def compute_action_exp(self, preds, internals, obs, available_actions):
        q_vals = self._get_qvals_from_pred(preds)
        batch_size = preds[self.action_keys[0]].shape[0]

        # reduce feature dim, build action_key dim
        actions = OrderedDict()
        actions_gpu = OrderedDict()
        values = []
        for key in self.action_keys:
            # random action across some environments based on the actors epsilon
            rand_mask = (
                (self.epsilon > torch.rand(batch_size)).nonzero().squeeze(-1)
            )
            action = self._action_from_q_values(q_vals[key])
            rand_act = torch.randint(
                self.action_space[key][0],
                (rand_mask.shape[0], 1),
                dtype=torch.long,
            ).to(action.device)
            action[rand_mask] = rand_act
            actions[key] = action.squeeze(1).cpu()
            actions_gpu[key] = action

            values.append(
                self._get_action_values(q_vals[key], action, batch_size)
            )

        values = self._values_to_tensor(values)
        internals = {k: torch.stack(vs) for k, vs in internals.items()}

        return actions, {"actions": actions_gpu, "internals": internals}

    @classmethod
    def _exp_spec(cls, exp_len, batch_sz, obs_space, act_space, internal_space):
        act_key_len = len(act_space.keys())
        internal_spec = {
            k: (exp_len, batch_sz, *shape)
            for k, shape in internal_space.items()
        }

        spec = {
            "actions": (exp_len, batch_sz, act_key_len, 1),
            "internals": internal_spec,
        }
        return spec

    def _get_qvals_from_pred(self, preds):
        return preds

    def _action_from_q_values(self, q_vals):
        return q_vals.argmax(dim=-1, keepdim=True)

    def _get_action_values(self, q_vals, action, batch_size=0):
        return q_vals.gather(1, action)

    def _values_to_tensor(self, values):
        return torch.cat(values, dim=1)