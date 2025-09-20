from typing import Optional, Sequence, Union
import os
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2axangle
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import cv2 as cv
from collections import deque
from typing import Dict
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
import json
from huggingface_hub import snapshot_download

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .openvla_utils import get_vla

ACTION_DIM = 7
PROPRIO_DIM = 8
NUM_ACTIONS_CHUNK = 8

def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    """
    Find a specific checkpoint file matching a pattern.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
        file_pattern: String pattern to match in filenames

    Returns:
        str: Path to the matching checkpoint file

    Raises:
        AssertionError: If no files or multiple files match the pattern
    """
    assert os.path.isdir(
        pretrained_checkpoint
    ), f"Checkpoint path must be a directory: {pretrained_checkpoint}"

    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            checkpoint_files.append(full_path)

    assert (
        len(checkpoint_files) == 1
    ), f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in directory: {pretrained_checkpoint}"

    return checkpoint_files[0]

def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a component's state dict from checkpoint and handle DDP prefix if present.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dict: The processed state dictionary for loading
    """
    state_dict = torch.load(checkpoint_path, weights_only=True)

    # If the component was trained with DDP, elements in the state dict have prefix "module." which we must remove
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    # fmt: on

class ProprioProjector(nn.Module):
    """
    Projects proprio state inputs into the LLM's embedding space.
    """
    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim

        self.fc1 = nn.Linear(self.proprio_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        # proprio: (bsz, proprio_dim)
        projected_features = self.fc1(proprio)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x

class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        action = self.model(rearranged_actions_hidden_states)
        return action

class OpenVLAInference:
    def __init__(
        self,
        saved_model_path: str = "fhliang/jaco_adv_500",
        unnorm_key: Optional[str] = "jaco_dataset",
        policy_setup: str = "jaco",
        horizon: int = 1,
        pred_action_horizon: int = 1,
        exec_horizon: int = 1,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 15
        elif policy_setup == "jaco":
            unnorm_key = "jaco_dataset" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 15
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        self._ckpt_path = snapshot_download(saved_model_path)

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.processor = AutoProcessor.from_pretrained(saved_model_path, trust_remote_code=True)
        # self.vla = AutoModelForVision2Seq.from_pretrained(
        #     saved_model_path,
        #     attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        #     torch_dtype=torch.bfloat16,
        #     low_cpu_mem_usage=True,
        #     trust_remote_code=True,
        # ).cuda()
        cfg = GenerateConfig(
            pretrained_checkpoint = saved_model_path, 
            use_l1_regression = True, 
            use_diffusion = False, 
            use_film = False, 
            num_images_in_input = 1, 
            use_proprio = True, 
            load_in_8bit = False, 
            load_in_4bit = False, 
            center_crop = True, 
            num_open_loop_steps = NUM_ACTIONS_CHUNK, 
            unnorm_key = self.unnorm_key, 
        )
        self.vla = get_vla(cfg)

        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task = None
        self.task_description = None
        self.num_image_history = 0

        # Initialize projector and move to device
        proprio_projector = ProprioProjector(
            llm_dim=self.vla.llm_dim,
            proprio_dim=PROPRIO_DIM,  # 8-dimensional proprio for LIBERO
        ).to("cuda:0")
        proprio_projector = proprio_projector.to(torch.bfloat16).to("cuda:0")
        proprio_projector.eval()
        
        # For me self._ckpt_path is
        # "/home/shihanzh/.cache/huggingface/hub/models--fhliang--jaco_adv_500/snapshots/11437747406ee0b2975b5247405c061bd0ea024e"
        checkpoint_path = find_checkpoint_file(
            self._ckpt_path, "proprio_projector"
        )
        state_dict = load_component_state_dict(checkpoint_path)
        proprio_projector.load_state_dict(state_dict)
        
        self.proprio_projector = proprio_projector
    
        dataset_statistics_path = os.path.join(
            self._ckpt_path, "dataset_statistics.json"
        )
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
            self.norm_stats = norm_stats

        self.vla.norm_stats = self.norm_stats

        action_head = L1RegressionActionHead(
            input_dim=self.vla.llm_dim, hidden_dim=self.vla.llm_dim, action_dim=ACTION_DIM
        )

        checkpoint_path = find_checkpoint_file(self._ckpt_path, "action_head")
        state_dict = load_component_state_dict(checkpoint_path)
        action_head.load_state_dict(state_dict)
        action_head = action_head.to(torch.bfloat16).to("cuda:0")
        action_head.eval()
        
        self.action_head = action_head

        self.action_queue = deque(maxlen=8)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.action_queue = deque(maxlen=8)

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, obs = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)

        image: Image.Image = Image.fromarray(image)

        if len(self.action_queue) == 0:
            # Collect all input images
                # all_images = [obs["full_image"]]
                # if cfg.num_images_in_input > 1:
                #     all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

                # # Process images
                # all_images = prepare_images_for_vla(all_images, cfg)

            # Extract primary image and additional images
            primary_image = image # all_images.pop(0)

            # Build VLA prompt
            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"

            # Process primary image
            inputs = self.processor(prompt, primary_image).to("cuda:0", dtype=torch.bfloat16)

            # Process additional wrist images if any
            # if all_images:
            #     all_wrist_inputs = [
            #         processor(prompt, image_wrist).to(DEVICE, dtype=torch.bfloat16)
            #         for image_wrist in all_images
            #     ]
            #     # Concatenate all images
            #     primary_pixel_values = inputs["pixel_values"]
            #     all_wrist_pixel_values = [
            #         wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs
            #     ]
            #     inputs["pixel_values"] = torch.cat(
            #         [primary_pixel_values] + all_wrist_pixel_values, dim=1
            #     )


            # Process proprioception data if used
            proprio = None
            # proprio = obs["agent"] # obs["state"]
            # proprio = np.concatenate([
            #     obs["extra"]["tcp_pose"].ravel(),
            #     obs["agent"]["controller"]["gripper"]["target_qpos"][:1]
            # ], axis=0)
            # FIXME: This should be just obs['agent']['eef_pos'].
            proprio = np.concatenate([obs['agent']['eef_pos'][:-1], [obs['agent']['qpos'][-1]]])
            # proprio = obs['agent']['qpos'].copy()

            proprio_norm_stats = self.norm_stats[self.unnorm_key]["proprio"]
            
            # normalize proprio {obs["state"] = normalize_proprio(proprio, proprio_norm_stats)}
            mask = proprio_norm_stats.get("mask", np.ones_like(proprio_norm_stats["min"], dtype=bool))
            proprio_high, proprio_low = np.array(proprio_norm_stats["max"]), np.array(
                proprio_norm_stats["min"]
            )
            
            normalized_proprio = np.clip(
                np.where(
                    mask,
                    2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
                    proprio,
                ),
                a_min=-1.0,
                a_max=1.0,
            )
            obs["state"] = normalized_proprio
            proprio = obs["state"]

            # Custom action head for continuous actions
            action, _ = self.vla.predict_action(
                **inputs,
                unnorm_key=self.unnorm_key, # self.unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=self.proprio_projector,
                action_head=self.action_head,
                # below are not used
                noisy_action_projector=None,
                use_film=False,
            )

            actions = [action[i] for i in range(len(action))]
            self.action_queue.extend(actions)

        # Get action from queue
        raw_actions = self.action_queue.popleft()[None]

        # predict action (7-dof; un-normalize for bridgev2)
        # inputs = self.processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        # raw_actions = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)[None]
        # # print(f"*** raw actions {raw_actions} ***")

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        
        elif self.policy_setup == "jaco":
            # action["gripper"] = 2.0 * raw_action["open_gripper"] + 1
            action["gripper"] = raw_action["open_gripper"]

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
