import numpy as np
from Wrapper.body_model_wrapper import BodyModelWrapper, MeshData
from SUPR.supr.pytorch.supr import SUPR
from utils import SMPLX_NAMES

class SUPR_WRAPPER(BodyModelWrapper):
    def preload_body_model(self, gender):
        self._model = SUPR(f'data/body_models/supr/supr_{gender}.npy')
        self.joint_names = list(SMPLX_NAMES)
        self.parameter_names = {'pose': list(SMPLX_NAMES), 'trans': ['pelvis']}
        self.joint_parents = np.asarray([-1] + self._model.parent.cpu().tolist())
        return self

    def _model_output(self, input_params, shape_params):
        # berechnung der neuen werte
        input_params = {k: v.reshape(1, -1) for k, v in input_params.items()}

        return self._model(
            betas=shape_params,
            #expression=self._body_exp_tensor,
            **input_params,
        )

    def ik_joints(self, input_params, shape_params=None, expression=None):
        return self._model_output(input_params, shape_params).joints[0]

    def forward(self, input_params, shape_params, expression=None):
        model_output = self._model_output(input_params, shape_params)
        verts = model_output.vertices[0].detach().numpy()
        joints = model_output.joints[0].detach().numpy()
        faces = self._model.faces

        return MeshData(verts, joints, faces)
