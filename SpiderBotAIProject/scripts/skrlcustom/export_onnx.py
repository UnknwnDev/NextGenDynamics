"""Export trained SpiderBot policy model to ONNX format.

Usage:
    python scripts/skrlcustom/export_onnx.py --checkpoint path/to/agent_30000.pt
"""

import argparse
import sys
from pathlib import Path

import gymnasium
import numpy as np
import torch

# Add agents dir directly to avoid IsaacLab/omni dependency from package __init__
AGENTS_DIR = Path(__file__).resolve().parents[2] / (
    "source/SpiderBotAIProject/SpiderBotAIProject"
    "/tasks/manager_based/spiderbot_ai/agents"
)
sys.path.insert(0, str(AGENTS_DIR))

from skrl_custom_ppo_model import SharedRecurrentModel
from onnx_wrapper import PolicyOnnxWrapper


def infer_dims_from_checkpoint(state_dict):
    """Infer model dimensions from checkpoint weight shapes."""
    obs_dim = state_dict["obs_encoder.0.weight"].shape[1]
    act_dim = state_dict["policy_layer.0.weight"].shape[0]
    hidden_size = state_dict["gru.weight_hh_l0"].shape[1]
    return obs_dim, act_dim, hidden_size


def make_mock_observation_space(obs_dim):
    """Create mock observation space required by SharedRecurrentModel constructor."""
    return gymnasium.spaces.Dict({
        "observations": gymnasium.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
        "height_data": gymnasium.spaces.Box(-np.inf, np.inf, shape=(1, 64, 64), dtype=np.float32),
        "bev_data": gymnasium.spaces.Box(-np.inf, np.inf, shape=(3, 64, 64), dtype=np.float32),
        "nav_data": gymnasium.spaces.Box(-np.inf, np.inf, shape=(1, 33, 33), dtype=np.float32),
    })


def make_mock_action_space(act_dim):
    """Create mock action space required by SharedRecurrentModel constructor."""
    return gymnasium.spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)


def export(checkpoint_path: Path, output_path: Path):
    device = torch.device("cpu")

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    policy_state = checkpoint["policy"]
    obs_dim, act_dim, hidden_size = infer_dims_from_checkpoint(policy_state)
    print(f"[INFO] Inferred dims: obs={obs_dim}, act={act_dim}, hidden={hidden_size}")

    # Reconstruct SharedRecurrentModel
    obs_space = make_mock_observation_space(obs_dim)
    act_space = make_mock_action_space(act_dim)

    model = SharedRecurrentModel(
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        num_envs=1,
    )
    model.load_state_dict(policy_state)
    model.eval()

    # Extract scaler mean/var from checkpoint
    preprocessor_state = checkpoint.get("state_preprocessor", {})
    scaler_mean = preprocessor_state["running_mean"].float().to(device)
    scaler_var = preprocessor_state["running_variance"].float().to(device)

    # Build ONNX wrapper
    wrapper = PolicyOnnxWrapper(model, scaler_mean, scaler_var)
    wrapper.eval()

    # Dummy inputs for tracing
    batch = 1
    dummy_inputs = (
        torch.randn(batch, obs_dim),
        torch.randn(batch, 1, 64, 64),
        torch.randn(batch, 3, 64, 64),
        torch.randn(batch, 1, 33, 33),
        torch.zeros(1, batch, hidden_size),
    )

    # Export
    print(f"[INFO] Exporting ONNX to: {output_path}")
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(output_path),
        input_names=["observations", "height_data", "bev_data", "nav_data", "gru_hidden"],
        output_names=["actions", "gru_hidden_out"],
        dynamic_axes={
            "observations": {0: "batch"},
            "height_data": {0: "batch"},
            "bev_data": {0: "batch"},
            "nav_data": {0: "batch"},
            "gru_hidden": {1: "batch"},
            "actions": {0: "batch"},
            "gru_hidden_out": {1: "batch"},
        },
    )
    print("[INFO] Done.")


def main():
    parser = argparse.ArgumentParser(description="Export SpiderBot policy to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to agent_*.pt checkpoint.")
    parser.add_argument("--output", type=Path, default=None, help="Output .onnx path (default: next to checkpoint).")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve() if args.output else checkpoint_path.parent / "spiderbot.onnx"

    export(checkpoint_path, output_path)


if __name__ == "__main__":
    main()
