from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np
import torch


class MeshData(NamedTuple):
    """A single posed mesh in viewer coordinates, with joints in model order."""

    vertices: np.ndarray
    joints: np.ndarray
    faces: np.ndarray


class BodyModelWrapper(ABC):
    def __init__(self):
        # Names describe output joints; parameter metadata is keyed by pose group.
        # Parents include the root (-1). Extra surface landmarks have no parent.
        self.joint_names = []
        self.joint_parents = np.empty(0, dtype=np.int64)
        self.parameter_names = {}
        # Groups without explicit limits use the viewer's rotation controls.
        self.parameter_limits = {}

    @torch.no_grad()
    def evaluate(self, input_params, shape_params=None, expression=None):
        """Evaluate for display without building an autograd graph."""
        return MeshData(*self.forward(input_params, shape_params, expression))

    def ik_joints(self, input_params, shape_params=None, expression=None):
        """Differentiable joint/landmark positions in viewer units, without ground offset."""
        raise NotImplementedError

    def ik_parameter_masks(self, input_params):
        """Optimize articulation while keeping global placement fixed."""
        masks = {}
        for name in self.parameter_names:
            if name not in input_params or name in ('trans', 'global_orient'):
                continue
            mask = torch.ones_like(input_params[name], dtype=torch.bool)
            if name == 'pose':
                mask[:, 0] = False  # STAR, SUPR and ANNY include root rotation.
            masks[name] = mask
        return masks

    @abstractmethod
    def preload_body_model(self, gender):
        """Load the model and populate its joint and parameter metadata."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, input_params, shape_params=None, expression=None):
        """Return MeshData without mutating inputs (tensor betas or ANNY phenotypes)."""
        raise NotImplementedError
