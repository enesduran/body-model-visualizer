"""Integration checks using real assets: python tests/ik_models.py."""

import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from main import AppWindow
from Wrapper.wrapper_dict import WRAPPER_CLASSES
from inverse_kinematics import solve_ik


def check_model(name):
    gender = AppWindow.BODY_MODEL_GENDERS[name][0]
    model = WRAPPER_CLASSES[name]().preload_body_model(gender)
    params = copy.deepcopy(AppWindow.POSE_PARAMS[name])
    params['trans'][0, 0] = torch.tensor([0.1, 0.2, -0.1])
    if 'global_orient' in params:
        params['global_orient'][0, 0, 1] = 0.1
    else:
        params['pose'][0, 0, 1] = 0.1
    shape = dict(AppWindow.ANNY_PHENOTYPE_DEFAULTS) if name == 'ANNY' else torch.full((1, AppWindow.BODY_MODEL_N_BETAS[name]), 0.05)
    expression = torch.zeros(1, 10)
    original = copy.deepcopy(params)
    with torch.no_grad():
        initial = model.ik_joints(params, shape, expression)
        displayed = model.evaluate(params, shape, expression).joints
    np.testing.assert_allclose(initial.numpy(), displayed, atol=2e-6)
    goal = copy.deepcopy(params)
    if name in ('SMPL', 'SMPLX'):
        goal['body_pose'][0, 15, 2] = 0.25
    elif name == 'MANO':
        goal['hand_pose'][0, 0, 2] = 0.25
    elif name == 'FLAME':
        goal['jaw_pose'][0, 0, 0] = 0.25
    elif name == 'MHR':
        goal['model_parameters'][0, 46] = 0.25
    else:
        goal['pose'][0, 1 if name == 'ANNY' else 16, 2] = 0.25
    with torch.no_grad():
        destination = model.ik_joints(goal, shape, expression)
    selected = int(torch.linalg.vector_norm(destination - initial, dim=-1).argmax())
    target = initial.clone()
    target[selected] = destination[selected]
    start = time.perf_counter()
    result = solve_ik(model, params, target, shape, expression, [selected])
    duration = time.perf_counter() - start
    print(f'{name}: joint {selected}, {result.initial_error:.6f} -> {result.final_error:.6f} m, '
          f'{result.iterations} steps, {duration:.2f}s', flush=True)
    assert result.initial_error > 0.001
    assert result.final_error < result.initial_error * 0.3, name
    masks = model.ik_parameter_masks(params)
    for key in params:
        torch.testing.assert_close(params[key], original[key])
        mask = masks.get(key, torch.zeros_like(params[key], dtype=torch.bool))
        torch.testing.assert_close(result.parameters[key][~mask], original[key][~mask])
    with torch.no_grad():
        fitted = model.ik_joints(result.parameters, shape, expression)
    assert torch.isfinite(fitted).all()
    np.testing.assert_allclose(torch.linalg.vector_norm(fitted[selected] - target[selected]).item(), result.final_error, atol=1e-6)


if __name__ == '__main__':
    for name in AppWindow.BODY_MODEL_NAMES:
        check_model(name)
    print('All eight IK model checks passed.', flush=True)
