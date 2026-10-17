"""Run with a desktop session and local model assets: python tests/gui_smoke.py."""

import sys
import faulthandler
import traceback
import time
import copy
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import AppWindow, gui


def render_frames(app, window):
    for _ in range(3):
        window.window.post_redraw()
        assert app.run_one_tick(), 'Test window closed before checks finished'
        # Layout runs on redraw; allow a frame interval between event-loop ticks.
        time.sleep(0.02)


def check_ik(window):
    model = window._active_model()
    name = window._body_model.selected_text
    params = window.pose_params[name]
    goal = copy.deepcopy(params)
    if name in ('SMPL', 'SMPLX'):
        goal['body_pose'][0, 15, 2] += 0.25
    elif name == 'MANO':
        goal['hand_pose'][0, 0, 2] += 0.25
    elif name == 'FLAME':
        goal['jaw_pose'][0, 0, 0] += 0.25
    elif name == 'MHR':
        goal['model_parameters'][0, 46] += 0.25
    else:
        goal['pose'][0, 1 if name == 'ANNY' else 16, 2] += 0.25
    with torch.no_grad():
        destination = model.ik_joints(goal, window.get_shape_params(), window._body_exp_tensor).numpy()
    destination += np.array([0., window.ground_offset or 0., 0.])
    selected = int(np.linalg.norm(destination - window.joints, axis=1).argmax())
    window.joints[selected] = destination[selected]
    result = window._on_run_ik()
    assert result is not None and result.initial_error > 0.001, name
    assert result.final_error < result.initial_error * 0.5, (name, result)

    pixels, depths, valid = window._project_joint_positions()
    frame = window._scene.frame
    valid &= (pixels[:, 0] > 0) & (pixels[:, 0] < frame.width)
    valid &= (pixels[:, 1] > 0) & (pixels[:, 1] < frame.height)
    candidates = np.flatnonzero(valid)
    assert len(candidates), name
    selected = int(candidates[len(candidates) // 2])
    x, y = pixels[selected]
    point = window._scene.scene.camera.unproject(float(x), float(y), float(depths[selected]), frame.width, frame.height)
    np.testing.assert_allclose(point, window.joints[selected], atol=2e-4)
    window.drag_key_pressed = True
    window._on_mouse_widget(SimpleNamespace(type=gui.MouseEvent.Type.BUTTON_DOWN,
                                           x=frame.x + x, y=frame.y + y))
    assert window.is_dragging, name
    # Exercise the real pressed-button DRAG event, without relying on Q focus.
    window.drag_key_pressed = False
    window._drag_joints.checked = True
    dragged = window.selected_joint
    before = window.joints[dragged].copy()
    window._on_mouse_widget(SimpleNamespace(type=gui.MouseEvent.Type.DRAG,
                                           x=frame.x + x + 8, y=frame.y + y))
    assert window._ik_target_dirty, name
    assert np.linalg.norm(window.joints[dragged] - before) > 1e-5, name
    window._on_mouse_widget(SimpleNamespace(type=gui.MouseEvent.Type.BUTTON_UP,
                                           x=frame.x + x, y=frame.y + y))
    assert not window.is_dragging and window.drag_depth is None
    window.drag_key_pressed = False
    window._drag_joints.checked = False
    print(f'GUI IK and joint picking passed: {name}', flush=True)


def check_number_edits(window):
    """Every parameter slider mirrors its value into the paired number field."""
    window._sync_pose_joint_sliders()
    window._sync_global_sliders()
    for name in ('_body_beta_val', '_body_exp_val', '_trans_x', '_trans_y', '_trans_z',
                 '_rot_x', '_rot_y', '_rot_z', '_transparency', '_body_pose_joint_val'):
        control = getattr(window, name)
        assert abs(control.edit.double_value - control.double_value) < 1e-6, name
    for name in ('_body_pose_joint_x', '_body_pose_joint_y', '_body_pose_joint_z'):
        control = getattr(window, name)
        assert control.edit.int_value == control.int_value, name
    # Typing a value moves the slider and fires the same callback as dragging it.
    before = window._trans_x.double_value
    window._trans_x._on_edit(before + 0.05)
    assert abs(window._trans_x.double_value - (before + 0.05)) < 1e-6
    assert abs(float(window.pose_params[window._body_model.selected_text]['trans'][0, 0, 0]) - (before + 0.05)) < 1e-6
    window._trans_x._on_edit(before)
    # Out-of-range input is clamped to the slider limits (MHR hides these sliders).
    if window._body_model.selected_text != 'MHR':
        window._body_pose_joint_x._on_edit(1000)
        assert window._body_pose_joint_x.int_value == 180
        window._body_pose_joint_x._on_edit(0)


def check_toolbar(window):
    toolbar, scene = window._toolbar.frame, window._scene.frame
    assert scene.x <= toolbar.x and toolbar.get_right() <= scene.get_right(), (toolbar, scene)
    assert scene.y <= toolbar.y and toolbar.get_bottom() <= scene.get_bottom(), (toolbar, scene)
    assert toolbar.width > 0 and toolbar.height > 0
    for direction in ([0., 0., 1.], [1., 0., 0.], [0., 1., 0.]):
        window._look_from(direction)
    window._on_camera_reset()


def check_models(window):
    for index, name in enumerate(window.body_model_names):
        window._body_model.selected_text = name
        window._on_body_model(name, index)
        window._show_joints.checked = True
        window._on_show_joints(True)
        window._show_joint_labels.checked = True
        for gender_index, gender in enumerate(window.body_model_genders[name]):
            window._body_model_gender.selected_text = gender
            window._on_body_model_gender(gender, gender_index)
            window._on_body_beta_val(0.1)
            if name == 'MHR':
                window._on_body_pose_joint_val(0.01)
            else:
                window._on_body_pose_joint_axis(0, 5)
            check_number_edits(window)
            assert window._scene.scene.has_geometry('__skeleton__')
            assert window._skeleton_button.is_on and window._labels_button.is_on
            assert len(window._joint_geometry_radii) == len(window.joints)
            assert len(window.joint_labels_3d_list) == len(window.joints)
            if gender_index == 0:
                check_ik(window)
            print(f'GUI callbacks passed: {name}-{gender}', flush=True)
            yield
        if name == 'ANNY':
            window._anny_phenotype.selected_text = 'height'
            window._on_anny_phenotype_val(0.6)
            window._on_anny_reset_shape()
        # Toolbar toggles drive the same state as the sidebar checkboxes.
        window._set_skeleton_visible(False)
        assert not window._show_joints.checked and not window._skeleton_button.is_on
        assert not window._joint_geometry_radii
        assert not window._scene.scene.has_geometry('__skeleton__')
        window._set_joint_labels_visible(False)
        assert not window._show_joint_labels.checked and not window._labels_button.is_on
        assert not window.joint_labels_3d_list
        window._set_drag_joints(True)
        assert window._drag_joints.checked and window._show_joints.checked
        assert window._drag_button.is_on and window._skeleton_button.is_on
        window._set_drag_joints(False)
        window._set_skeleton_visible(False)
        assert not window._drag_button.is_on and not window._skeleton_button.is_on
        yield

    window._body_model.selected_text = 'SMPL'
    window._on_body_model('SMPL', 0)
    window._on_run_ik()


def main():
    app = gui.Application.instance
    app.initialize()
    window = AppWindow(1280, 800)
    failed = False
    try:
        render_frames(app, window)
        assert window._scene.frame.get_right() <= window._settings_panel.frame.x, (window._scene.frame, window._settings_panel.frame)
        assert window._settings_panel.frame.width >= 25 * window.em
        check_toolbar(window)
        window._on_menu_toggle_settings_panel()
        render_frames(app, window)
        assert window._scene.frame.width == window.window.content_rect.width, (window._scene.frame, window.window.content_rect, window._settings_panel.visible)
        check_toolbar(window)
        window._on_menu_toggle_settings_panel()
        render_frames(app, window)
        for _ in check_models(window):
            # Exercise every tab with each model's conditional controls.
            for tab_index in range(2):
                window._settings_panel.selected_tab_index = tab_index
                window.window.set_needs_layout()
                render_frames(app, window)
                assert window._scene.frame.get_right() <= window._settings_panel.frame.x, (window._scene.frame, window._settings_panel.frame)
        print('All GUI smoke checks passed.', flush=True)
    except Exception:
        failed = True
        traceback.print_exc()
    finally:
        # Open3D on macOS can crash if quit() is called inside a tick callback.
        # Close between event-loop iterations instead.
        app.quit()
        app.run_one_tick()
    return int(failed)


if __name__ == '__main__':
    faulthandler.enable(file=sys.stderr)
    sys.exit(main())
