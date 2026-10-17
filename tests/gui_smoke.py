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
    params = AppWindow.POSE_PARAMS[name]
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
    destination += np.array([0., AppWindow.GROUND_OFFSET or 0., 0.])
    selected = int(np.linalg.norm(destination - AppWindow.JOINTS, axis=1).argmax())
    AppWindow.JOINTS[selected] = destination[selected]
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
    np.testing.assert_allclose(point, AppWindow.JOINTS[selected], atol=2e-4)
    AppWindow.DRAG_KEY_PRESSED = True
    window._on_mouse_widget(SimpleNamespace(type=gui.MouseEvent.Type.BUTTON_DOWN,
                                           x=frame.x + x, y=frame.y + y))
    assert AppWindow.IS_DRAGGING, name
    # Exercise the real pressed-button DRAG event, without relying on Q focus.
    AppWindow.DRAG_KEY_PRESSED = False
    window._drag_joints.checked = True
    dragged = AppWindow.SELECTED_JOINT
    before = AppWindow.JOINTS[dragged].copy()
    window._on_mouse_widget(SimpleNamespace(type=gui.MouseEvent.Type.DRAG,
                                           x=frame.x + x + 8, y=frame.y + y))
    assert window._ik_target_dirty, name
    assert np.linalg.norm(AppWindow.JOINTS[dragged] - before) > 1e-5, name
    window._on_mouse_widget(SimpleNamespace(type=gui.MouseEvent.Type.BUTTON_UP,
                                           x=frame.x + x, y=frame.y + y))
    assert not AppWindow.IS_DRAGGING and AppWindow.DRAG_DEPTH is None
    AppWindow.DRAG_KEY_PRESSED = False
    window._drag_joints.checked = False
    print(f'GUI IK and joint picking passed: {name}', flush=True)


def check_models(window):
    for index, name in enumerate(AppWindow.BODY_MODEL_NAMES):
        window._body_model.selected_text = name
        window._on_body_model(name, index)
        window._show_joints.checked = True
        window._on_show_joints(True)
        window._show_joint_labels.checked = True
        for gender_index, gender in enumerate(AppWindow.BODY_MODEL_GENDERS[name]):
            window._body_model_gender.selected_text = gender
            window._on_body_model_gender(gender, gender_index)
            window._on_body_beta_val(0.1)
            if name == 'MHR':
                window._on_body_pose_joint_val(0.01)
            else:
                window._on_body_pose_joint_x(5)
            assert window._scene.scene.has_geometry('__skeleton__')
            assert len(window._joint_geometry_radii) == len(AppWindow.JOINTS)
            assert len(window.joint_labels_3d_list) == len(AppWindow.JOINTS)
            if gender_index == 0:
                check_ik(window)
            print(f'GUI callbacks passed: {name}-{gender}', flush=True)
            yield
        if name == 'ANNY':
            window._anny_phenotype.selected_text = 'height'
            window._on_anny_phenotype_val(0.6)
            window._on_anny_reset_shape()
        window._show_joints.checked = False
        window._on_show_joints(False)
        assert not window._joint_geometry_radii
        assert not window._scene.scene.has_geometry('__skeleton__')
        window._show_joint_labels.checked = False
        window._on_show_joint_labels(False)
        assert not window.joint_labels_3d_list
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
        window._on_menu_toggle_settings_panel()
        render_frames(app, window)
        assert window._scene.frame.width == window.window.content_rect.width, (window._scene.frame, window.window.content_rect, window._settings_panel.visible)
        window._on_menu_toggle_settings_panel()
        render_frames(app, window)
        for _ in check_models(window):
            # Exercise every tab with each model's conditional controls.
            for tab_index in range(3):
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
