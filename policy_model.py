"""ResNet18 (ImageNet-pretrained) + current-gripper-aperture -> primitive-id policy.

The gripper scalar is necessary, not optional: `primitives.to_action(primitive,
current_gripper)`'s mapping for "keep"-type primitives depends on the current
gripper state, so a primitive id alone under-specifies the resulting 7-D mujoco
action. See POLICY_TRAINING.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from policy_data import NUM_CLASSES

GRIPPER_EMBED_DIM = 16


def imagenet_transform():
    return torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()


class PolicyNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.gripper_embed = nn.Sequential(
            nn.Linear(1, GRIPPER_EMBED_DIM), nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(self.feature_dim + GRIPPER_EMBED_DIM, num_classes)

    def forward(self, image: torch.Tensor, gripper: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(image)
        g = self.gripper_embed(gripper)
        return self.head(torch.cat([feats, g], dim=-1))
