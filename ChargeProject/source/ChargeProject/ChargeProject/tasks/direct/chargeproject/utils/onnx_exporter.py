import torch
import torch.nn as nn

from ChargeProject.tasks.direct.chargeproject.agents.skrl_custom_ppo_model import SharedRecurrentModel

class OnnxStepPolicyLSTM(nn.Module):
    def __init__(self, model: SharedRecurrentModel):
        super().__init__()
        self.m = model  # SharedRecurrentModel 인스턴스

    def forward(self,
                observations: torch.Tensor,   # (B, obs_dim)
                height_data: torch.Tensor,     # (B, 64, 64)
                bev_data: torch.Tensor,        # (B, 3, 64, 64)
                h0: torch.Tensor,              # (L, B, H)
                c0: torch.Tensor):             # (L, B, H)

        # encoders
        obs_encoded = self.m.obs_encoder(observations)                 # (B, 256)
        height_encoded = self.m.height_encoder(height_data.unsqueeze(1))  # (B, 128)
        bev_encoded = self.m.bev_encoder(bev_data)                     # (B, 128)

        fused = torch.cat([obs_encoded, height_encoded, bev_encoded], dim=-1)  # (B, 512)

        # LSTM one-step (seq_len=1)
        x = fused.unsqueeze(1)  # (B, 1, 512)  batch_first=True
        y, (h1, c1) = self.m.lstm(x, (h0, c0))  # y: (B, 1, hidden)
        y = y.squeeze(1)  # (B, hidden)

        # final MLP + policy head
        z = self.m.net(y)                              # (B, 128)
        action_mean = torch.tanh(self.m.policy_layer(z))  # (B, act_dim)

        return action_mean, h1, c1


def export_onnx(model: SharedRecurrentModel):
    model.eval()
    wrapper = OnnxStepPolicyLSTM(model).eval().cpu()

    B = 1
    obs_dim = model.observation_space["observations"].shape[0]
    act_dim = model.num_actions
    L = model.num_layers
    H = model.hidden_size

    dummy_observations = torch.zeros(B, obs_dim)
    dummy_height = torch.zeros(B, 64, 64)
    dummy_bev = torch.zeros(B, 3, 64, 64)
    dummy_h0 = torch.zeros(L, B, H)
    dummy_c0 = torch.zeros(L, B, H)

    torch.onnx.export(
        wrapper,
        (dummy_observations, dummy_height, dummy_bev, dummy_h0, dummy_c0),
        "spider_bot_v0.onnx",
        opset_version=17,
        input_names=["observations", "height_data", "bev_data", "h0", "c0"],
        output_names=["action_mean", "h1", "c1"],
        dynamic_axes={
            "observations": {0: "batch"},
            "height_data": {0: "batch"},
            "bev_data": {0: "batch"},
            "h0": {1: "batch"},
            "c0": {1: "batch"},
            "action_mean": {0: "batch"},
            "h1": {1: "batch"},
            "c1": {1: "batch"},
        }
    )
