import unittest

import numpy as np
import torch

from inverse_kinematics import solve_ik


class BoundedModel:
    parameter_limits = {'pose': np.array([[-0.5, 0.5], [0., 0.]])}

    def ik_parameter_masks(self, params):
        return {'pose': torch.tensor([[True, False]])}

    def ik_joints(self, params, shape=None, expression=None):
        x, fixed = params['pose'][0]
        zero = x * 0
        root = torch.stack((zero, zero, zero))
        endpoint = torch.stack((x, fixed, zero))
        return torch.stack((root, endpoint)) + params['trans']


class IKTests(unittest.TestCase):
    def setUp(self):
        self.model = BoundedModel()
        self.params = {'pose': torch.zeros(1, 2), 'trans': torch.tensor([[1., 2., 3.]])}

    def test_unreachable_target_respects_limits_and_preserves_inputs(self):
        target = self.model.ik_joints(self.params).detach()
        target[1, 0] += 2.
        result = solve_ik(self.model, self.params, target)
        self.assertLess(result.final_error, result.initial_error)
        self.assertLessEqual(result.parameters['pose'][0, 0], 0.5)
        self.assertGreaterEqual(result.parameters['pose'][0, 0], -0.5)
        self.assertEqual(result.parameters['pose'][0, 1], 0.)
        torch.testing.assert_close(result.parameters['trans'], self.params['trans'])
        torch.testing.assert_close(self.params['pose'], torch.zeros(1, 2))

    def test_unchanged_targets_are_a_noop(self):
        result = solve_ik(self.model, self.params, self.model.ik_joints(self.params))
        self.assertEqual(result.iterations, 0)
        torch.testing.assert_close(result.parameters['pose'], self.params['pose'])

    def test_immovable_root_does_not_change_pose(self):
        target = self.model.ik_joints(self.params).detach()
        target[0, 0] += 0.2
        result = solve_ik(self.model, self.params, target)
        self.assertAlmostEqual(result.final_error, result.initial_error)
        torch.testing.assert_close(result.parameters['pose'], self.params['pose'])

    def test_invalid_targets_are_rejected(self):
        target = self.model.ik_joints(self.params).detach()
        target[1, 0] = float('nan')
        with self.assertRaises(ValueError):
            solve_ik(self.model, self.params, target)
        with self.assertRaises(ValueError):
            solve_ik(self.model, self.params, torch.zeros(9, 3))

    def test_solver_works_inside_display_no_grad_context(self):
        with torch.no_grad():
            target = self.model.ik_joints(self.params).detach()
            target[1, 0] += 0.2
            result = solve_ik(self.model, self.params, target)
        self.assertLess(result.final_error, 0.002)
        self.assertFalse(result.parameters['pose'].requires_grad)


if __name__ == '__main__':
    unittest.main()
