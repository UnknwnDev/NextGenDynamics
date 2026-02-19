# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space


def _unwrap_policy_observation_space(observation_space):
    """Unwrap nested Dict observation spaces down to the policy input terms.

    Isaac Lab manager-based environments commonly return nested Dict spaces (e.g. top-level "policy").
    We need to reach the Dict that contains the actual term keys expected by the model (e.g. "observations").
    """
    current = observation_space

    for _ in range(8):  # avoid infinite loops on unexpected/recursive structures
        spaces = None

        if hasattr(current, "spaces") and isinstance(getattr(current, "spaces"), Mapping):
            spaces = current.spaces
        elif isinstance(current, Mapping):
            spaces = current

        if not isinstance(spaces, Mapping) or not spaces:
            break

        # Stop once we reach the Dict that contains the expected term keys.
        if "observations" in spaces:
            break

        # Prefer the common Isaac Lab convention.
        if "policy" in spaces:
            current = spaces["policy"]
            continue

        # Fallback: unwrap single-key wrappers (e.g. {"states": {...}}).
        if len(spaces) == 1:
            current = next(iter(spaces.values()))
            continue

        break

    return current


class SharedRecurrentModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, num_envs, init_log_std=0.0):
        observation_space = _unwrap_policy_observation_space(observation_space)

        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20.0,
            max_log_std=2.0,
            reduction="sum",
            role="policy",
        )
        DeterministicMixin.__init__(self, clip_actions=False, role="value")

        obs_dim = self.observation_space["observations"].shape[0]
        height_dim = self.observation_space["height_data"].shape
        bev_dim = self.observation_space["bev_data"].shape
        nav_dim = self.observation_space["nav_data"].shape

        assert height_dim in [(64, 64), (1, 64, 64)], f"Unexpected height_data shape: {height_dim}"
        assert bev_dim == (3, 64, 64), f"Unexpected bev_data shape: {bev_dim}"
        assert nav_dim in [(1, 33, 33), (33, 33)], f"Unexpected nav_data shape: {nav_dim}"

        act_dim = self.num_actions
        self.num_envs = num_envs

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Height encoder CNN (64x64 -> 128)
        self.height_encoder = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.GroupNorm(2, 8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 8 -> 4
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 4 -> 2
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Flatten(),  # 32 * 2 * 2 = 128
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # BEV encoder CNN (3x64x64 -> 128)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.GroupNorm(2, 8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 8 -> 4
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 4 -> 2
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Flatten(),  # 32 * 2 * 2 = 128
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Nav encoder CNN ((1, 33, 33) -> 128)
        self.nav_encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),  # 33 -> 17
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),  # 17 -> 9
            nn.GroupNorm(2, 8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 9 -> 5
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 5 -> 3
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 3 -> 2
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Flatten(),  # 32 * 2 * 2 = 128
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.num_layers = 1
        self.hidden_size = 512
        self.sequence_length = 32

        self.gru = nn.GRU(
            input_size=320,  # 128 (obs) + 64 (height) + 64 (bev) + 64 (nav)
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.net = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Identity()
        )

        self.policy_layer = nn.Sequential(
            nn.Linear(128, act_dim),
            nn.Tanh()
        )
        self.value_layer = nn.Linear(128, 1)

        self.log_std_parameter = nn.Parameter(
            torch.full(size=(self.num_actions,), fill_value=float(init_log_std)),
            requires_grad=True,
        )

        self._shared_output = None

    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (self.num_layers, self.num_envs, self.hidden_size),  # GRU hidden state
                ],
            }
        }

    def act(self, inputs, role):
        if role == "policy":
            self._shared_output = None
            return GaussianMixin.act(self, inputs, role)
        if role == "value":
            return DeterministicMixin.act(self, inputs, role)
        raise ValueError(f"Unknown role '{role}'")

    def compute(self, inputs, role=""):
        if self._shared_output is None:
            states = unflatten_tensorized_space(self.observation_space, inputs["states"])
            observations = states["observations"]
            height_data = states["height_data"]
            bev_data = states["bev_data"]
            nav_data = states["nav_data"]

            terminated = inputs.get("terminated", None)

            # Encode observations
            obs_encoded = self.obs_encoder(observations)

            if height_data.dim() == 3:
                height_data = height_data.unsqueeze(1)
            height_encoded = self.height_encoder(height_data)

            bev_encoded = self.bev_encoder(bev_data)

            if nav_data.dim() == 3:
                nav_data = nav_data.unsqueeze(1)
            nav_encoded = self.nav_encoder(nav_data)

            # Fusion
            fused = torch.cat([obs_encoded, height_encoded, bev_encoded, nav_encoded], dim=-1)

            # GRU rollout
            rnn_output, rnn_dict = self.gru_rollout(self.gru, fused, terminated, inputs["rnn"])

            # Final layers
            net = self.net(rnn_output)

            self._shared_output = net, rnn_dict

        if role == "policy":
            net, rnn_dict = self._shared_output
            mean = torch.tanh(self.policy_layer(net))
            return mean, self.log_std_parameter, rnn_dict

        if role == "value":
            net, rnn_dict = self._shared_output
            self._shared_output = None
            output = self.value_layer(net)
            return output, rnn_dict

        raise ValueError(f"Unknown role '{role}'")

    def gru_rollout(self, model, states, terminated, hidden_states):
        if self.training:
            # reshape to (batch, seq, features)
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])

            hidden_states[0] = hidden_states[0].view(
                self.num_layers, -1, self.sequence_length, hidden_states[0].shape[-1]
            )[:, :, 0, :].contiguous()

            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )
                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states[0] = model(rnn_input[:, i0:i1, :], hidden_states[0])
                    hidden_states[0][:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, hidden_states[0] = model(rnn_input, hidden_states[0])
        else:
            # evaluation mode: one step at a time
            rnn_input = states.view(-1, 1, states.shape[-1])
            hidden_states[0] = hidden_states[0].contiguous()
            rnn_output, hidden_states[0] = model(rnn_input, hidden_states[0])

        # flatten batch + sequence
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        return rnn_output, {"rnn": hidden_states}
