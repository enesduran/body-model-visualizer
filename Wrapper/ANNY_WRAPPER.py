import os
import sys
import roma
import torch
import numpy as np

ANNY_SRC_DIR = os.path.join(os.path.dirname(__file__), "anny", "src")
if os.path.isdir(ANNY_SRC_DIR) and ANNY_SRC_DIR not in sys.path:
    sys.path.insert(0, ANNY_SRC_DIR)

from anny.models.full_model import create_model

from Wrapper.body_model_wrapper import BodyModelWrapper, MeshData


class ANNY_WRAPPER(BodyModelWrapper):
    def preload_body_model(self, gender):
        self._model = create_model()
        self.joint_names = list(self._model.bone_labels)
        self.parameter_names = {'pose': list(self.joint_names), 'trans': []}
        self.joint_parents = np.asarray(self._model.bone_parents, dtype=np.int64)
        self._faces = self._model.get_triangular_faces().cpu().numpy().astype(np.int32)
        return self

    def _model_output(self, input_params, shape_params):
        rotations = roma.rotvec_to_rotmat(input_params['pose'][0])
        pose = roma.Rigid(
            rotations, torch.zeros((len(rotations), 3), dtype=torch.float64)
        )[None].to_homogeneous()
        return self._model(
            pose_parameters=pose, phenotype_kwargs=shape_params or {},
            local_changes_kwargs={}, pose_parameterization=None, return_bone_ends=False,
        )

    def _viewer_positions(self, positions, input_params):
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
