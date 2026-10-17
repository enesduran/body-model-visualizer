from pathlib import Path

import torch
import numpy as np
from mhr.mhr import MHR
from Wrapper.body_model_wrapper import BodyModelWrapper, MeshData
from paths import body_model_dir

NUM_IDENTITY_BLENDSHAPES = 45
NUM_FACE_EXPRESSION_BLENDSHAPES = 72
NUM_MODEL_PARAMETERS = 204
LOD = 1
MHR_ASSETS_DIR = Path(body_model_dir("mhr")) / "assets"
# Values beyond this are treated as unbounded and clamped to a usable range.
UNBOUNDED_SENTINEL = 1.0e30
UNBOUNDED_FALLBACK = 3.14

class MHR_WRAPPER(BodyModelWrapper):
    def preload_body_model(self, gender=None):
        if not MHR_ASSETS_DIR.is_dir():
            raise FileNotFoundError(f"MHR assets not found at {MHR_ASSETS_DIR}")
        self._model = MHR.from_files(
            folder=MHR_ASSETS_DIR, device=torch.device("cpu"), lod=LOD
        )
        self._identity_coeffs = torch.zeros(1, NUM_IDENTITY_BLENDSHAPES)
        self._face_expr_coeffs = torch.zeros(1, NUM_FACE_EXPRESSION_BLENDSHAPES)
        self._faces = np.asarray(self._model.character.mesh.faces, dtype=np.int32)

        pt = self._model.character.parameter_transform
        self.parameter_names['model_parameters'] = list(pt.names[:NUM_MODEL_PARAMETERS])
        self.joint_names = list(self._model.character.skeleton.joint_names)
        self.joint_parents = np.asarray(
            self._model.character.skeleton.joint_parents, dtype=np.int64
        )

        mins, maxs = self._model.character.model_parameter_limits
        mins = np.asarray(mins[:NUM_MODEL_PARAMETERS], dtype=np.float32)
        maxs = np.asarray(maxs[:NUM_MODEL_PARAMETERS], dtype=np.float32)
        mins = np.where(mins < -UNBOUNDED_SENTINEL, -UNBOUNDED_FALLBACK, mins)
        maxs = np.where(maxs > UNBOUNDED_SENTINEL, UNBOUNDED_FALLBACK, maxs)
        self.parameter_limits['model_parameters'] = np.stack([mins, maxs], axis=1)

        return self

    def forward(self, input_params, shape_params=None, expression=None):
        if shape_params is not None:
            n = min(shape_params.shape[1], NUM_IDENTITY_BLENDSHAPES)
            self._identity_coeffs[0, :n] = shape_params[0, :n]

        # Face Expression falls übergeben
        if expression is not None:
        #if 'expression' in input_params:
        #    exp = input_params['expression']
            n = min(expression.shape[1], NUM_FACE_EXPRESSION_BLENDSHAPES)
            self._face_expr_coeffs[0, :n] = expression[0, :n]
            
        model_parameters = self._pose_parameters(input_params)

        with torch.no_grad():
            verts, skel_state = self._model(
                self._identity_coeffs,
                model_parameters,
                self._face_expr_coeffs,
            )

        verts = verts.squeeze(0).detach().numpy() / 100.0
        joints = skel_state.squeeze(0)[:, :3].detach().numpy() / 100.0
        faces = self._faces

        return MeshData(verts, joints, faces)

    def _pose_parameters(self, input_params):
        parameters = input_params['model_parameters'].clone()
        if 'trans' in input_params:
            parameters[:, :3] = input_params['trans'].reshape(1, 3)
        if 'global_orient' in input_params:
            parameters[:, 3:6] = input_params['global_orient'].reshape(1, 3)
        return parameters

    def ik_parameter_masks(self, input_params):
        # MHR's flexiblePose set changes bone lengths; it is not articulation.
        sets = self._model.character.parameter_transform.parameter_sets
        fixed = np.zeros(NUM_MODEL_PARAMETERS, dtype=bool)
        for name in ('rigid', 'scale', 'handScale', 'flexiblePose'):
            fixed |= np.asarray(sets[name])[:NUM_MODEL_PARAMETERS]
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
