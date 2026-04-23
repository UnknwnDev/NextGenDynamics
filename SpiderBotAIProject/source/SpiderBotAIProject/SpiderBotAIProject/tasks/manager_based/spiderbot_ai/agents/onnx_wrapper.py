import torch
import torch.nn as nn


class PolicyOnnxWrapper(nn.Module):
    """ONNX-exportable wrapper around SharedRecurrentModel's policy inference path.

    Bakes in RunningStandardScaler normalization so raw observations can be fed directly.
    Exposes GRU hidden state as explicit input/output tensors.
    """

    def __init__(self, model, scaler_mean, scaler_var, scaler_epsilon=1e-8, scaler_clip=5.0):
        super().__init__()

        self.obs_encoder = model.obs_encoder
        self.height_encoder = model.height_encoder
        self.bev_encoder = model.bev_encoder
        self.nav_encoder = model.nav_encoder
        self.gru = model.gru
        self.net = model.net
        self.policy_layer = model.policy_layer

        # Bake-in scaler parameters
        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_var", scaler_var)
        self.scaler_epsilon = scaler_epsilon
        self.scaler_clip = scaler_clip

        obs_dim = model.obs_encoder[0].in_features
        self.split_sizes = [12288, 4096, 1089, obs_dim]

    def forward(self, observations, height_data, bev_data, nav_data, gru_hidden):
        flat = torch.cat([
            bev_data.reshape(bev_data.shape[0], -1),
            height_data.reshape(height_data.shape[0], -1),
            nav_data.reshape(nav_data.shape[0], -1),
            observations,
        ], dim=-1)

        # RunningStandardScaler normalization
        flat = torch.clamp(
            (flat - self.scaler_mean) / (torch.sqrt(self.scaler_var) + self.scaler_epsilon),
            min=-self.scaler_clip,
            max=self.scaler_clip,
        )

        bev_flat, height_flat, nav_flat, obs_flat = torch.split(flat, self.split_sizes, dim=-1)

        obs_encoded = self.obs_encoder(obs_flat)
        height_encoded = self.height_encoder(height_flat.reshape(-1, 1, 64, 64))
        bev_encoded = self.bev_encoder(bev_flat.reshape(-1, 3, 64, 64))
        nav_encoded = self.nav_encoder(nav_flat.reshape(-1, 1, 33, 33))

        fused = torch.cat([obs_encoded, height_encoded, bev_encoded, nav_encoded], dim=-1)

        # GRU (single-step inference)
        rnn_input = fused.unsqueeze(1)
        rnn_output, gru_hidden_out = self.gru(rnn_input, gru_hidden)
        rnn_output = rnn_output.squeeze(1)

        # Shared net -> policy head
        net_out = self.net(rnn_output)
        actions = self.policy_layer(net_out)

        return actions, gru_hidden_out
