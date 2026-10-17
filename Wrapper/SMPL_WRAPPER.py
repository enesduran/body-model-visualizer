import numpy as np
from smplx import SMPL, SMPLX, MANO, FLAME

from Wrapper.body_model_wrapper import BodyModelWrapper, MeshData
from paths import body_model_dir
from utils import (
    SMPL_NAMES, SMPLX_NAMES, MANO_NAMES, FLAME_KEYPOINT_NAMES,
    smpl_joint_names, smplx_body_joint_names, hand_joint_names,
)


class SMPL_WRAPPER(BodyModelWrapper):
    """Adapt the SMPL family to the same display interface as other models."""

    MODEL_CLASSES = {'SMPL': SMPL, 'SMPLX': SMPLX, 'MANO': MANO, 'FLAME': FLAME}
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
        options = {'gender': gender}
        if self.model_type != 'SMPL':
            options.update(use_pca=False, flat_hand_mean=True, use_face_contour=True)
        self._model = self.MODEL_CLASSES[self.model_type](
            body_model_dir(self.model_type.lower()), **options
        )
        self.joint_names = list(self.JOINT_NAMES[self.model_type])
        self.parameter_names = {
            group: list(names) for group, names in self.PARAMETER_NAMES[self.model_type].items()
        }
        self.joint_parents = self._model.parents.detach().cpu().numpy().copy()
        self._faces = np.asarray(self._model.faces, dtype=np.int32)
        return self

    @property
    def ik_model(self):
        """Differentiable model used by the existing SMPL/SMPL-X IK solver."""
        return self._model

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
