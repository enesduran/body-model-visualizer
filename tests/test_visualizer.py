import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from main import AppWindow, gui
from Wrapper.body_model_wrapper import BodyModelWrapper, MeshData


class FakeScene:
    def __init__(self):
        self.geometries = {}
        self.transforms = {}
        self.labels = {}
        self.additions = 0
        self.next_label = 0

    def has_geometry(self, name):
        return name in self.geometries

    def add_geometry(self, name, geometry, material):
        self.geometries[name] = geometry
        self.additions += 1

    def remove_geometry(self, name):
        del self.geometries[name]
        self.transforms.pop(name, None)

    def modify_geometry_material(self, name, material):
        assert name in self.geometries

    def set_geometry_transform(self, name, transform):
        self.transforms[name] = transform.copy()

    def add_3d_label(self, position, name):
        handle = self.next_label
        self.next_label += 1
        self.labels[handle] = name
        return handle

    def remove_3d_label(self, handle):
        # Reject stale handles as well as double removal.
        del self.labels[handle]


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.original_joints = AppWindow.JOINTS
        self.original_selected = AppWindow.SELECTED_JOINT
        AppWindow.JOINTS = np.array([[0., 0., 0.], [0., 1., 0.], [0., 2., 0.]])
        AppWindow.SELECTED_JOINT = None
        self.window = AppWindow.__new__(AppWindow)
        self.scene = FakeScene()
        self.window._scene = SimpleNamespace(
            scene=self.scene, add_3d_label=self.scene.add_3d_label,
            remove_3d_label=self.scene.remove_3d_label,
        )
        self.window._body_model = SimpleNamespace(selected_text='SMPL')
        self.window._show_joint_labels = SimpleNamespace(checked=False)
        self.window.joint_labels_3d_list = []
        self.window._joint_geometry_radii = {}
        self.model = SimpleNamespace(
            joint_names=['pelvis', 'left_hip', 'left_knee'],
            joint_parents=np.array([-1, 0, 1]),
        )
        self.window._active_model = lambda: self.model

    def tearDown(self):
        AppWindow.JOINTS = self.original_joints
        AppWindow.SELECTED_JOINT = self.original_selected

    def test_labels_do_not_accumulate_or_remove_stale_handles(self):
        for _ in range(5):
            self.window._on_show_joint_labels(True)
            self.assertEqual(len(self.scene.labels), 3)
            self.assertEqual(len(self.window.joint_labels_3d_list), 3)
        self.window._on_show_joint_labels(False)
        self.window._on_show_joint_labels(False)
        self.assertFalse(self.scene.labels)
        self.assertFalse(self.window.joint_labels_3d_list)
        AppWindow.JOINTS = None
        self.window._on_show_joint_labels(True)

    def test_pose_updates_reuse_spheres_at_absolute_positions(self):
        self.window._on_show_joints(True, update_transparency=False)
        spheres = [self.scene.geometries[f'__joints_{i}__'] for i in range(3)]
        for sphere in spheres:
            np.testing.assert_allclose(sphere.get_center(), 0, atol=1e-12)
        for _ in range(2):
            AppWindow.JOINTS += np.array([1., 0., 0.])
            self.window._on_show_joints(True, update_transparency=False)
            for i, sphere in enumerate(spheres):
                self.assertIs(self.scene.geometries[f'__joints_{i}__'], sphere)
                np.testing.assert_allclose(
                    self.scene.transforms[f'__joints_{i}__'][:3, 3], AppWindow.JOINTS[i]
                )
        # Three spheres once, plus one skeleton per refresh.
        self.assertEqual(self.scene.additions, 6)
        np.testing.assert_array_equal(
            np.asarray(self.scene.geometries['__skeleton__'].points), AppWindow.JOINTS
        )

    def test_switching_joint_count_and_radius_then_hiding_clears_geometry(self):
        self.window._on_show_joints(True, update_transparency=False)
        old = self.scene.geometries['__joints_1__']
        # Another model can assign a different marker size to the same index.
        self.model.joint_names[1] = 'nose'
        AppWindow.JOINTS = AppWindow.JOINTS[:2]
        self.window._on_show_joints(True, update_transparency=False)
        self.assertNotIn('__joints_2__', self.scene.geometries)
        self.assertIsNot(old, self.scene.geometries['__joints_1__'])
        self.window._on_show_joints(False, update_transparency=False)
        self.window._on_show_joints(False, update_transparency=False)
        self.assertFalse(self.scene.geometries)
        self.assertFalse(self.window._joint_geometry_radii)

    def test_invalid_parent_indices_and_extra_landmarks(self):
        self.model.joint_parents = [-1, 0, 1, 999, 4]
        np.testing.assert_array_equal(self.window._skeleton_edges(7), [[0, 1], [1, 2]])
        self.assertEqual(self.window._skeleton_edges(0).shape, (0, 2))

    def test_anny_shape_callbacks_use_selected_variant_and_reset_ground(self):
        calls = []
        self.window._body_model.selected_text = 'ANNY'
        self.window._body_model_gender = SimpleNamespace(selected_text='default')
        self.window._anny_phenotype = SimpleNamespace(selected_text='height')
        self.window._anny_phenotype_val = SimpleNamespace(double_value=0.)
        self.window._anny_phenotype_values = dict(AppWindow.ANNY_PHENOTYPE_DEFAULTS)
        self.window.load_body_model = lambda *args, **kwargs: calls.append((args, kwargs))
        previous_ground = AppWindow.GROUND_OFFSET
        try:
            AppWindow.GROUND_OFFSET = 2.
            self.window._on_anny_phenotype_val(0.7)
            self.assertIsNone(AppWindow.GROUND_OFFSET)
            self.assertEqual(self.window._anny_phenotype_values['height'], 0.7)
            self.window._on_anny_reset_shape()
            self.assertEqual(calls, [(('ANNY',), {'gender': 'default'})] * 2)
        finally:
            AppWindow.GROUND_OFFSET = previous_ground


class InteractionTests(unittest.TestCase):
    def setUp(self):
        names = ('JOINTS', 'SELECTED_JOINT', 'IS_DRAGGING', 'DRAG_DEPTH', 'DRAG_KEY_PRESSED')
        self.saved = {name: getattr(AppWindow, name) for name in names}
        AppWindow.JOINTS = np.zeros((1, 3))
        AppWindow.SELECTED_JOINT = None
        AppWindow.IS_DRAGGING = False
        AppWindow.DRAG_DEPTH = None
        AppWindow.DRAG_KEY_PRESSED = True
        self.window = AppWindow.__new__(AppWindow)
        self.window.em = 16
        self.window._ik_target_dirty = False
        self.window._show_joints = SimpleNamespace(checked=True)
        self.window._drag_joints = SimpleNamespace(checked=False)
        self.window._scene = SimpleNamespace(
            frame=SimpleNamespace(x=100, y=50, width=800, height=600),
            scene=SimpleNamespace(camera=SimpleNamespace(unproject=Mock(return_value=[1., 2., 3.]))),
            set_view_controls=Mock(),
        )
        self.window._project_joint_positions = lambda: (np.array([[50., 40.]]), np.array([0.5]), np.array([True]))
        self.window._current_joint_display_names = lambda: ['joint']
        self.window._on_run_ik = Mock()
        self.window._on_show_joints = Mock()
        self.window._update_label = Mock()

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(AppWindow, name, value)

    def mouse(self, event_type):
        return self.window._on_mouse_widget(SimpleNamespace(type=event_type, x=150, y=90))

    def test_drag_release_solves_once_for_every_model(self):
        for model in AppWindow.BODY_MODEL_NAMES:
            with self.subTest(model=model):
                self.window._body_model = SimpleNamespace(selected_text=model)
                AppWindow.DRAG_KEY_PRESSED = True
                self.window._on_run_ik.reset_mock()
                self.mouse(gui.MouseEvent.Type.BUTTON_DOWN)
                self.mouse(gui.MouseEvent.Type.DRAG)
                self.window._scene.scene.camera.unproject.assert_called_with(50, 40, 0.5, 800, 600)
                self.mouse(gui.MouseEvent.Type.BUTTON_UP)
                self.window._on_key_widget(SimpleNamespace(type=gui.KeyEvent.Type.UP, key=gui.KeyName.Q))
                self.window._on_run_ik.assert_called_once()
                self.assertFalse(AppWindow.IS_DRAGGING)
                self.assertIsNone(AppWindow.DRAG_DEPTH)

    def test_releasing_q_before_mouse_also_solves_once(self):
        self.mouse(gui.MouseEvent.Type.BUTTON_DOWN)
        self.mouse(gui.MouseEvent.Type.DRAG)
        self.window._on_key_widget(SimpleNamespace(type=gui.KeyEvent.Type.UP, key=gui.KeyName.Q))
        self.mouse(gui.MouseEvent.Type.BUTTON_UP)
        self.window._on_run_ik.assert_called_once()

    def test_drag_mode_works_without_keyboard_focus(self):
        AppWindow.DRAG_KEY_PRESSED = False
        self.window._drag_joints.checked = True
        self.mouse(gui.MouseEvent.Type.BUTTON_DOWN)
        self.mouse(gui.MouseEvent.Type.DRAG)
        np.testing.assert_allclose(AppWindow.JOINTS[0], [1., 2., 3.])
        self.assertTrue(self.window._ik_target_dirty)
        self.mouse(gui.MouseEvent.Type.BUTTON_UP)
        self.window._on_run_ik.assert_called_once()

    def test_keyboard_nudge_applies_only_on_key_down(self):
        AppWindow.SELECTED_JOINT = 0
        self.window._on_key_widget(SimpleNamespace(type=gui.KeyEvent.Type.DOWN, key=gui.KeyName.TWO))
        self.window._on_key_widget(SimpleNamespace(type=gui.KeyEvent.Type.UP, key=gui.KeyName.TWO))
        np.testing.assert_allclose(AppWindow.JOINTS[0], [0.01, 0, 0])
        self.assertTrue(self.window._ik_target_dirty)


class GradientTests(unittest.TestCase):
    def test_display_evaluation_disables_gradients_without_changing_inputs(self):
        class Model(BodyModelWrapper):
            def preload_body_model(self, gender):
                return self

            def forward(self, input_params, shape_params=None, expression=None):
                self.grad_enabled = torch.is_grad_enabled()
                result = (input_params['pose'] * 2).numpy()
                return MeshData(result, result, np.empty((0, 3), dtype=np.int32))

        model = Model()
        params = {'pose': torch.ones(1, 3, requires_grad=True)}
        original = params['pose'].detach().clone()
        output = model.evaluate(params)
        self.assertIsInstance(output, MeshData)
        self.assertFalse(model.grad_enabled)
        self.assertTrue(torch.is_grad_enabled())
        torch.testing.assert_close(params['pose'], original)


if __name__ == '__main__':
    unittest.main()
