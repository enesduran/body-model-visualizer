"""Adapters that give every supported body model the same interface.

Each adapter imports its model package inside ``preload_body_model`` so that a
missing or broken optional dependency (anny, mhr, STAR, SUPR) only makes that
model unavailable instead of breaking the import of this module.
"""
import os
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from paths import body_model_dir
from utils import (
    SMPL_NAMES, SMPLX_NAMES, MANO_NAMES, STAR_NAMES, FLAME_KEYPOINT_NAMES,
    smpl_joint_names, smplx_body_joint_names, hand_joint_names,
)


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


# --------------------------------------------------------------------------- #
# SMPL family: SMPL, SMPL-X, MANO, FLAME (smplx package)
# --------------------------------------------------------------------------- #
class SMPLFamilyWrapper(BodyModelWrapper):
    MODEL_TYPES = ('SMPL', 'SMPLX', 'MANO', 'FLAME')
    JOINT_NAMES = {
        'SMPL': SMPL_NAMES, 'SMPLX': SMPLX_NAMES,
        'MANO': MANO_NAMES, 'FLAME': FLAME_KEYPOINT_NAMES,
    }
    PARAMETER_NAMES = {
        'SMPL': {'global_orient': ['root'], 'body_pose': smpl_joint_names},
        'SMPLX': {
            'global_orient': ['root'], 'body_pose': smplx_body_joint_names,
            'left_hand_pose': hand_joint_names, 'right_hand_pose': hand_joint_names,
            'jaw_pose': ['jaw'], 'leye_pose': ['leye'], 'reye_pose': ['reye'],
        },
        'MANO': {'global_orient': ['root'], 'hand_pose': hand_joint_names},
        'FLAME': {
            'global_orient': ['root'], 'jaw_pose': ['jaw'], 'neck_pose': ['neck'],
            'leye_pose': ['leye'], 'reye_pose': ['reye'],
        },
    }

    def __init__(self, model_type):
        super().__init__()
        self.model_type = model_type

    def preload_body_model(self, gender):
        import smplx

        options = {'gender': gender}
        if self.model_type != 'SMPL':
            options.update(use_pca=False, flat_hand_mean=True, use_face_contour=True)
        model_class = getattr(smplx, self.model_type)
        self._model = model_class(body_model_dir(self.model_type.lower()), **options)
        self.joint_names = list(self.JOINT_NAMES[self.model_type])
        self.parameter_names = {
            group: list(names) for group, names in self.PARAMETER_NAMES[self.model_type].items()
        }
        self.joint_parents = self._model.parents.detach().cpu().numpy().copy()
        self._faces = np.asarray(self._model.faces, dtype=np.int32)
        return self

    def _model_output(self, input_params, shape_params=None, expression=None):
        params = {key: value.reshape(1, -1) for key, value in input_params.items()}
        if 'trans' in params:
            params['transl'] = params.pop('trans')
        if self.model_type in ('SMPLX', 'FLAME'):
            params['expression'] = expression
        return self._model(betas=shape_params, **params)

    def ik_joints(self, input_params, shape_params=None, expression=None):
        return self._model_output(input_params, shape_params, expression).joints[0]

    def forward(self, input_params, shape_params=None, expression=None):
        output = self._model_output(input_params, shape_params, expression)
        return MeshData(
            output.vertices[0].detach().cpu().numpy(),
            output.joints[0].detach().cpu().numpy(),
            self._faces,
        )


# --------------------------------------------------------------------------- #
# SUPR and STAR share the SMPL-style (betas, pose, trans) call convention
# --------------------------------------------------------------------------- #
class _PoseTransWrapper(BodyModelWrapper):
    """Common evaluation for models called as model(betas=..., pose=..., trans=...)."""

    def _model_output(self, input_params, shape_params):
        input_params = {k: v.reshape(1, -1) for k, v in input_params.items()}
        return self._model(betas=shape_params, **input_params)

    def ik_joints(self, input_params, shape_params=None, expression=None):
        return self._model_output(input_params, shape_params).joints[0]

    def forward(self, input_params, shape_params, expression=None):
        output = self._model_output(input_params, shape_params)
        return MeshData(
            output.vertices[0].detach().numpy(),
            output.joints[0].detach().numpy(),
            self._model.faces,
        )


class SUPRWrapper(_PoseTransWrapper):
    def preload_body_model(self, gender):
        from SUPR.supr.pytorch.supr import SUPR

        self._model = SUPR(os.path.join(body_model_dir('supr'), f'supr_{gender}.npy'))
        self.joint_names = list(SMPLX_NAMES)
        self.parameter_names = {'pose': list(SMPLX_NAMES), 'trans': ['pelvis']}
        self.joint_parents = np.asarray([-1] + self._model.parent.cpu().tolist())
        return self


class STARWrapper(_PoseTransWrapper):
    def preload_body_model(self, gender):
        from STAR.star.pytorch.star import STAR  # model paths come from STAR/star/config.py

        self._model = STAR(gender=gender.lower())
        self.joint_names = list(SMPL_NAMES)
        self.parameter_names = {'pose': list(STAR_NAMES), 'trans': ['pelvis']}
        self.joint_parents = np.asarray([-1] + self._model.parent.cpu().tolist())
        return self


# --------------------------------------------------------------------------- #
# ANNY (naver/anny): phenotype dict as shape, per-bone rotation vectors as pose
# --------------------------------------------------------------------------- #
class ANNYWrapper(BodyModelWrapper):
    def preload_body_model(self, gender):
        from anny.models.full_model import create_model

        self._model = create_model()
        self.joint_names = list(self._model.bone_labels)
        self.parameter_names = {'pose': list(self.joint_names), 'trans': []}
        self.joint_parents = np.asarray(self._model.bone_parents, dtype=np.int64)
        self._faces = self._model.get_triangular_faces().cpu().numpy().astype(np.int32)
        return self

    def _model_output(self, input_params, shape_params):
        import roma

        rotations = roma.rotvec_to_rotmat(input_params['pose'][0])
        pose = roma.Rigid(
            rotations, torch.zeros((len(rotations), 3), dtype=torch.float64)
        )[None].to_homogeneous()
        return self._model(
            pose_parameters=pose, phenotype_kwargs=shape_params or {},
            local_changes_kwargs={}, pose_parameterization=None, return_bone_ends=False,
        )

    def _viewer_positions(self, positions, input_params):
        import roma

        # ANNY is Z-up; rotate into the viewer's Y-up frame, then apply translation.
        rotation = roma.euler_to_rotmat('x', [270.], degrees=True).to(positions)
        positions = positions @ rotation.T
        if 'trans' in input_params:
            positions = positions + input_params['trans'][0, 0].to(positions)
        return positions

    def ik_joints(self, input_params, shape_params=None, expression=None):
        output = self._model_output(input_params, shape_params)
        return self._viewer_positions(output['bone_poses'][0, :, :3, 3], input_params)

    def forward(self, input_params, shape_params=None, expression=None):
        output = self._model_output(input_params, shape_params)
        vertices = self._viewer_positions(output['vertices'][0], input_params)
        joints = self._viewer_positions(output['bone_poses'][0, :, :3, 3], input_params)
        return MeshData(
            vertices.detach().cpu().numpy(), joints.detach().cpu().numpy(), self._faces
        )


# --------------------------------------------------------------------------- #
# MHR (facebookresearch/MHR): scalar rig parameters with native limits
# --------------------------------------------------------------------------- #
class MHRWrapper(BodyModelWrapper):
    NUM_IDENTITY_BLENDSHAPES = 45
    NUM_FACE_EXPRESSION_BLENDSHAPES = 72
    NUM_MODEL_PARAMETERS = 204
    LOD = 1
    # Values beyond this are treated as unbounded and clamped to a usable range.
    UNBOUNDED_SENTINEL = 1.0e30
    UNBOUNDED_FALLBACK = 3.14

    def preload_body_model(self, gender=None):
        from mhr.mhr import MHR

        assets_dir = Path(body_model_dir('mhr')) / 'assets'
        if not assets_dir.is_dir():
            raise FileNotFoundError(f'MHR assets not found at {assets_dir}')
        self._model = MHR.from_files(folder=assets_dir, device=torch.device('cpu'), lod=self.LOD)
        self._identity_coeffs = torch.zeros(1, self.NUM_IDENTITY_BLENDSHAPES)
        self._face_expr_coeffs = torch.zeros(1, self.NUM_FACE_EXPRESSION_BLENDSHAPES)
        self._faces = np.asarray(self._model.character.mesh.faces, dtype=np.int32)

        n = self.NUM_MODEL_PARAMETERS
        pt = self._model.character.parameter_transform
        self.parameter_names['model_parameters'] = list(pt.names[:n])
        self.joint_names = list(self._model.character.skeleton.joint_names)
        self.joint_parents = np.asarray(self._model.character.skeleton.joint_parents, dtype=np.int64)

        mins, maxs = self._model.character.model_parameter_limits
        mins = np.asarray(mins[:n], dtype=np.float32)
        maxs = np.asarray(maxs[:n], dtype=np.float32)
        mins = np.where(mins < -self.UNBOUNDED_SENTINEL, -self.UNBOUNDED_FALLBACK, mins)
        maxs = np.where(maxs > self.UNBOUNDED_SENTINEL, self.UNBOUNDED_FALLBACK, maxs)
        self.parameter_limits['model_parameters'] = np.stack([mins, maxs], axis=1)
        return self

    def _pose_parameters(self, input_params):
        parameters = input_params['model_parameters'].clone()
        if 'trans' in input_params:
            parameters[:, :3] = input_params['trans'].reshape(1, 3)
        if 'global_orient' in input_params:
            parameters[:, 3:6] = input_params['global_orient'].reshape(1, 3)
        return parameters

    def forward(self, input_params, shape_params=None, expression=None):
        if shape_params is not None:
            n = min(shape_params.shape[1], self.NUM_IDENTITY_BLENDSHAPES)
            self._identity_coeffs[0, :n] = shape_params[0, :n]
        if expression is not None:
            n = min(expression.shape[1], self.NUM_FACE_EXPRESSION_BLENDSHAPES)
            self._face_expr_coeffs[0, :n] = expression[0, :n]

        with torch.no_grad():
            verts, skel_state = self._model(
                self._identity_coeffs, self._pose_parameters(input_params), self._face_expr_coeffs,
            )
        # MHR works in centimetres.
        verts = verts.squeeze(0).detach().numpy() / 100.0
        joints = skel_state.squeeze(0)[:, :3].detach().numpy() / 100.0
        return MeshData(verts, joints, self._faces)

    def ik_parameter_masks(self, input_params):
        # MHR's flexiblePose set changes bone lengths; it is not articulation.
        n = self.NUM_MODEL_PARAMETERS
        sets = self._model.character.parameter_transform.parameter_sets
        fixed = np.zeros(n, dtype=bool)
        for name in ('rigid', 'scale', 'handScale', 'flexiblePose'):
            fixed |= np.asarray(sets[name])[:n]
        limits = self.parameter_limits['model_parameters']
        mask = ~fixed & (limits[:, 0] < limits[:, 1])
        return {'model_parameters': torch.as_tensor(mask).reshape_as(input_params['model_parameters'])}

    def ik_joints(self, input_params, shape_params=None, expression=None):
        # The MHR skeleton depends on rig parameters, not mesh blendshapes.
        parameters = self._pose_parameters(input_params)
        padding_size = self._model.character.parameter_transform.size - parameters.shape[1]
        padded = torch.cat((parameters, parameters.new_zeros((1, padding_size))), dim=1)
        character = self._model.character_torch
        joint_parameters = character.model_parameters_to_joint_parameters(padded)
        state = character.joint_parameters_to_skeleton_state(joint_parameters)
        return state[0, :, :3] / 100.0


# Model name -> factory. Preloading raises ImportError when the model's
# package is missing, which the app reports and skips.
WRAPPER_CLASSES = {
    **{name: partial(SMPLFamilyWrapper, name) for name in SMPLFamilyWrapper.MODEL_TYPES},
    'SUPR': SUPRWrapper,
    'STAR': STARWrapper,
    'ANNY': ANNYWrapper,
    'MHR': MHRWrapper,
}
