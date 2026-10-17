# This script is modified from https://github.com/isl-org/Open3D/blob/master/examples/python/gui/vis-gui.py

# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------
import os
import sys
import glob
import inspect
import torch
import joblib
import platform
import argparse
import numpy as np
import open3d as o3d
from loguru import logger
import open3d.visualization.gui as gui
import scipy.spatial.transform.rotation as R
import open3d.visualization.rendering as rendering
from collections import namedtuple
from functools import partial

# Model files are addressed relative to the project root ("data/body_models/...").
# A PyInstaller bundle unpacks them next to the code, so run from there and keep
# a log file, since a windowed app has no terminal.
if getattr(sys, 'frozen', False):
    os.chdir(getattr(sys, '_MEIPASS', os.path.dirname(sys.executable)))
    logger.add(os.path.join(os.path.expanduser('~'), 'Library', 'Logs',
                            'BodyModelVisualizer.log')
               if platform.system() == 'Darwin' else 'BodyModelVisualizer.log',
               rotation='5 MB', retention=3)

from utils import (
    get_checkerboard_plane,
    LEFT_HAND_KEYPOINT_NAMES,
    RIGHT_HAND_KEYPOINT_NAMES,
    HEAD_KEYPOINT_NAMES,
    FOOT_KEYPOINT_NAMES,
)
from inverse_kinematics import solve_ik
from paths import external_models_dir
from body_models import WRAPPER_CLASSES
from viewer_settings import Settings, ViewerSettingsPanel


isMacOS = (platform.system() == "Darwin")


class SliderWithEdit:
    """A slider paired with a number field that shows and edits the same value."""

    def __init__(self, slider_type, em, decimals=3):
        self._is_int = slider_type == gui.Slider.INT
        self.slider = gui.Slider(slider_type)
        self.edit = gui.NumberEdit(
            gui.NumberEdit.INT if self._is_int else gui.NumberEdit.DOUBLE)
        if not self._is_int:
            self.edit.decimal_precision = decimals
        self.edit.set_preferred_width(int(4.5 * em))
        self.widget = gui.Horiz(0.25 * em)
        self.widget.add_child(self.slider)
        self.widget.add_child(self.edit)
        self._callback = None
        self.slider.set_on_value_changed(self._on_slider)
        self.edit.set_on_value_changed(self._on_edit)

    def _on_slider(self, value):
        self._show(value)
        if self._callback is not None:
            self._callback(value)

    def _on_edit(self, value):
        value = min(max(value, self.slider.get_minimum_value),
                    self.slider.get_maximum_value)
        if self._is_int:
            value = int(round(value))
            self.slider.int_value = value
        else:
            self.slider.double_value = float(value)
        self._show(value)
        if self._callback is not None:
            self._callback(value)

    def _show(self, value):
        if self._is_int:
            self.edit.int_value = int(round(value))
        else:
            self.edit.double_value = float(value)

    def set_on_value_changed(self, callback):
        self._callback = callback

    def set_limits(self, low, high):
        self.slider.set_limits(low, high)
        self.edit.set_limits(low, high)

    @property
    def double_value(self):
        return self.slider.double_value

    @double_value.setter
    def double_value(self, value):
        self.slider.double_value = float(value)
        self._show(value)

    @property
    def int_value(self):
        return self.slider.int_value

    @int_value.setter
    def int_value(self, value):
        self.slider.int_value = int(value)
        self._show(value)

    @property
    def visible(self):
        return self.widget.visible

    @visible.setter
    def visible(self, value):
        self.widget.visible = value


def enable_chumpy_compatibility():
    """Keep legacy SMPL pickle files loadable on Python 3.11+ and NumPy 2."""
    if not hasattr(inspect, "getargspec"):
        inspect.ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")
        inspect.getargspec = lambda func: inspect.ArgSpec(
            *inspect.getfullargspec(func)[:4]
        )

    legacy_numpy_aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in legacy_numpy_aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


enable_chumpy_compatibility()


def euler_to_rotvec_for_body_model(body_model, euler_angle):
    if body_model.lower() == 'anny':
        euler_angle = [euler_angle[0], euler_angle[2], euler_angle[1]]
    return R.Rotation.from_euler('xyz', euler_angle, degrees=True).as_rotvec()


def rotvec_to_euler_for_body_model(body_model, rotvec):
    # Inverse of euler_to_rotvec_for_body_model, used to redisplay a stored
    # axis-angle pose value back onto the euler-angle UI sliders?
    euler = R.Rotation.from_rotvec(rotvec).as_euler('xyz', degrees=True)
    if body_model.lower() == 'anny':
        euler = [euler[0], euler[2], euler[1]]
    return euler


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SAVE = 4
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    # for mouse dragging joint feature
    IS_DRAGGING = False
    DRAG_DEPTH = None
    DRAG_KEY_PRESSED = False


    ## ANNY TYPES
    # interactive demo auswahlmöglichkeiten
    ANNY_MODEL_TYPES = ["default", "left hand", "right hand", "head", "notoes_collapse10pc", "notoes_collapse5pc"]
    ANNY_RIGS = [
        'default',
        'mixamo',
        'default-noeyes-notongue-noexpression-nobreasts-notoes',
        'default-noeyes-notongue-noexpression-nobreasts-notoes-nohands',
    ]

    # add Star and Anny
    BODY_MODEL_NAMES = ["SMPL", "SMPLX", "MANO", "FLAME", "SUPR", "STAR", "ANNY", "MHR"]
    # For ANNY the rig variants take the place of genders.
    BODY_MODEL_GENDERS = {
        'SMPL': ['neutral', 'male', 'female'],
        'SMPLX': ['neutral', 'male', 'female'],
        'MANO': ['neutral'],
        'FLAME': ['neutral', 'male', 'female'],
        'SUPR': ['neutral', 'male', 'female'],
        'STAR': ['neutral', 'male', 'female'],
        'ANNY': ['default'],
        'MHR': ['default']
    }

    BODY_MODEL_N_BETAS = {
        'SMPL': 10,
        'SMPLX': 10,
        'MANO': 10,
        'FLAME': 10,
        'SUPR': 10,
        'STAR': 10,
        'ANNY': 6, # ?? gender, age, muscle, weight, height,proportions
        'MHR': 45
    }
    ANNY_PHENOTYPE_NAMES = ["gender", "age", "muscle", "weight", "height", "proportions"]
    ANNY_PHENOTYPE_DEFAULTS = {
        "gender": 0.5,
        "age": 0.5,
        "muscle": 0.5,
        "weight": 0.5,
        "height": 0.5,
        "proportions": 0.5,
    }
    ANNY_REGION_ORDER = [
        "Root",
        "Torso",
        "Left Leg",
        "Right Leg",
        "Left Arm",
        "Right Arm",
        "Head & Face",
        "Tongue",
        "Other",
    ]

    # ANNY uses its own Blender-style bone names. Keep every known ANNY bone
    # in exactly one explicit display-size category, rather than deriving a
    # category from a substring in its name.
    ANNY_HAND_JOINT_NAMES = [
        'wrist.L',
        'finger1-1.L', 'finger1-2.L', 'finger1-3.L', 'metacarpal1.L',
        'finger2-1.L', 'finger2-2.L', 'finger2-3.L', 'metacarpal2.L',
        'finger3-1.L', 'finger3-2.L', 'finger3-3.L', 'metacarpal3.L',
        'finger4-1.L', 'finger4-2.L', 'finger4-3.L', 'metacarpal4.L',
        'finger5-1.L', 'finger5-2.L', 'finger5-3.L',
        'wrist.R',
        'finger1-1.R', 'finger1-2.R', 'finger1-3.R', 'metacarpal1.R',
        'finger2-1.R', 'finger2-2.R', 'finger2-3.R', 'metacarpal2.R',
        'finger3-1.R', 'finger3-2.R', 'finger3-3.R', 'metacarpal3.R',
        'finger4-1.R', 'finger4-2.R', 'finger4-3.R', 'metacarpal4.R',
        'finger5-1.R', 'finger5-2.R', 'finger5-3.R',
    ]
    ANNY_FOOT_JOINT_NAMES = [
        'foot.L',
        'toe1-1.L', 'toe1-2.L',
        'toe2-1.L', 'toe2-2.L', 'toe2-3.L',
        'toe3-1.L', 'toe3-2.L', 'toe3-3.L',
        'toe4-1.L', 'toe4-2.L', 'toe4-3.L',
        'toe5-1.L', 'toe5-2.L', 'toe5-3.L',
        'foot.R',
        'toe1-1.R', 'toe1-2.R',
        'toe2-1.R', 'toe2-2.R', 'toe2-3.R',
        'toe3-1.R', 'toe3-2.R', 'toe3-3.R',
        'toe4-1.R', 'toe4-2.R', 'toe4-3.R',
        'toe5-1.R', 'toe5-2.R', 'toe5-3.R',
    ]
    ANNY_HEAD_JOINT_NAMES = [
        'neck01', 'neck02', 'neck03', 'head', 'jaw',
        'special04', 'oris02', 'oris01', 'oris06.L', 'oris07.L',
        'oris06.R', 'oris07.R',
        'tongue00', 'tongue01', 'tongue02', 'tongue03', 'tongue04',
        'tongue07.L', 'tongue07.R', 'tongue06.L', 'tongue06.R',
        'tongue05.L', 'tongue05.R',
        'levator02.L', 'levator03.L', 'levator04.L', 'levator05.L',
        'levator02.R', 'levator03.R', 'levator04.R', 'levator05.R',
        'special01', 'oris04.L', 'oris03.L', 'oris04.R', 'oris03.R',
        'oris06', 'oris05', 'special03', 'levator06.L', 'levator06.R',
        'special06.L', 'special05.L', 'eye.L',
        'orbicularis03.L', 'orbicularis04.L',
        'special06.R', 'special05.R', 'eye.R',
        'orbicularis03.R', 'orbicularis04.R',
        'temporalis01.L', 'oculi02.L', 'oculi01.L',
        'temporalis01.R', 'oculi02.R', 'oculi01.R',
        'temporalis02.L', 'risorius02.L', 'risorius03.L',
        'temporalis02.R', 'risorius02.R', 'risorius03.R',
    ]
    ANNY_BODY_JOINT_NAMES = [
        'root',
        'pelvis.L', 'upperleg01.L', 'upperleg02.L',
        'lowerleg01.L', 'lowerleg02.L',
        'pelvis.R', 'upperleg01.R', 'upperleg02.R',
        'lowerleg01.R', 'lowerleg02.R',
        'spine05', 'spine04', 'spine03', 'spine02', 'breast.L', 'breast.R',
        'spine01',
        'clavicle.L', 'shoulder01.L', 'upperarm01.L', 'upperarm02.L',
        'lowerarm01.L', 'lowerarm02.L',
        'clavicle.R', 'shoulder01.R', 'upperarm01.R', 'upperarm02.R',
        'lowerarm01.R', 'lowerarm02.R',
    ]

    # MHR joint names come directly from the MHR skeleton. Keep the small-joint
    # groups explicit so their display size does not depend on name heuristics.
    MHR_HAND_JOINT_NAMES = [
        'r_wrist_twist', 'r_wrist',
        'r_pinky0', 'r_pinky1', 'r_pinky2', 'r_pinky3', 'r_pinky_null',
        'r_ring1', 'r_ring2', 'r_ring3', 'r_ring_null',
        'r_middle1', 'r_middle2', 'r_middle3', 'r_middle_null',
        'r_index1', 'r_index2', 'r_index3', 'r_index_null',
        'r_thumb0', 'r_thumb1', 'r_thumb2', 'r_thumb3', 'r_thumb_null',
        'l_wrist_twist', 'l_wrist',
        'l_pinky0', 'l_pinky1', 'l_pinky2', 'l_pinky3', 'l_pinky_null',
        'l_ring1', 'l_ring2', 'l_ring3', 'l_ring_null',
        'l_middle1', 'l_middle2', 'l_middle3', 'l_middle_null',
        'l_index1', 'l_index2', 'l_index3', 'l_index_null',
        'l_thumb0', 'l_thumb1', 'l_thumb2', 'l_thumb3', 'l_thumb_null',
    ]
    MHR_FOOT_JOINT_NAMES = [
        'l_foot', 'l_talocrural', 'l_subtalar', 'l_transversetarsal', 'l_ball',
        'r_foot', 'r_talocrural', 'r_subtalar', 'r_transversetarsal', 'r_ball',
    ]
    MHR_HEAD_JOINT_NAMES = [
        'c_neck', 'c_neck_twist1_proc', 'c_neck_twist0_proc',
        'c_head', 'c_head_null', 'c_jaw', 'c_jaw_null', 'c_teeth',
        'c_tongue0', 'c_tongue1', 'c_tongue2', 'c_tongue3', 'c_tongue4',
        'r_eye', 'r_eye_null', 'l_eye', 'l_eye_null',
    ]
    CAM_FIRST = True
    # Saved for the camera reset button.
    CAM_BOUNDS = None
    CAM_CENTER = None

    PRELOADED_BODY_MODELS = {}

    # save these values per model so ui can be representative of actual values
    BODY_BETAS = {}
    BODY_EXPS = {}
    TRANSPARENCY = {}

    POSE_PARAMS = {
        'SMPL': {
            'body_pose': torch.zeros(1, 23, 3),
            'global_orient': torch.zeros(1, 1, 3),
            'trans': torch.zeros(1, 1, 3), 
        },
        'SMPLX': {
            'body_pose': torch.zeros(1, 21, 3),
            'global_orient': torch.zeros(1, 1, 3),
            'left_hand_pose': torch.zeros(1, 15, 3),
            'right_hand_pose': torch.zeros(1, 15, 3),
            'jaw_pose': torch.zeros(1, 1, 3),
            'leye_pose': torch.zeros(1, 1, 3),
            'reye_pose': torch.zeros(1, 1, 3),
            'trans': torch.zeros(1, 1, 3), 
        },
        'MANO': {
            'hand_pose': torch.zeros(1, 15, 3),
            'global_orient': torch.zeros(1, 1, 3),
            'trans': torch.zeros(1, 1, 3),
        },
        'FLAME': {
            'global_orient': torch.zeros(1, 1, 3),
            'jaw_pose': torch.zeros(1, 1, 3),
            'neck_pose': torch.zeros(1, 1, 3),
            'leye_pose': torch.zeros(1, 1, 3),
            'reye_pose': torch.zeros(1, 1, 3),
            'trans': torch.zeros(1, 1, 3),
        },
        'SUPR': {
            'pose': torch.zeros(1, 75, 3),
            'trans': torch.zeros(1,1, 3),
        },
        'STAR' : {
            'pose': torch.zeros(1, 24, 3),
            #'betas': torch.zeros(1, 10), # not sure!!!!!
            'trans': torch.zeros(1, 1, 3),
        },
        'ANNY' : {
            'pose': torch.zeros(1, 163, 3),
            'trans': torch.zeros(1, 1, 3),
        },
        'MHR' : {
            'model_parameters': torch.zeros(1, 204),
            'global_orient': torch.zeros(1, 1, 3),
            'trans': torch.zeros(1, 1, 3),
        }
    }

    # Populated from wrapper metadata when models are loaded.
    JOINT_NAMES = {}
    KEYPOINT_NAMES = {}

    JOINTS = None
    SELECTED_JOINT = None
    BODY_TRANSL = None
    # GROUND_OFFSET is computed on the first render and cached
    # so that rotating the body does not move its root.
    GROUND_OFFSET = None

    def __init__(self, width, height):
        # Model selection during construction already refreshes the scene.
        self.joint_labels_3d_list = []
        self._joint_geometry_radii = {}
        self._ik_target_dirty = False
        self.window = gui.Application.instance.create_window(
            "Body Model Visualizer", width, height)
        w = self.window  # to make the code more concise

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        # ---- Settings panel ----
        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
        em = w.theme.font_size
        self.em = em

        # View, lighting and material controls (Open3D viewer boilerplate).
        self.viewer = ViewerSettingsPanel(self._scene, em)
        self.settings = self.viewer.settings

        self._settings_panel = gui.TabControl()
        self.model_settings = gui.ScrollableVert(
            0.5 * em, gui.Margins(0.75 * em, 0.75 * em, 0.75 * em, 0.75 * em))
        self._display_settings = gui.ScrollableVert(
            0.5 * em, gui.Margins(0.75 * em, 0.75 * em, 0.75 * em, 0.75 * em))
        self._settings_panel.add_tab("Model", self.model_settings)
        self._settings_panel.add_tab("Display", self._display_settings)
        self._settings_panel.set_on_selected_tab_changed(
            lambda index: self.window.set_needs_layout())

        def section(title, is_open=True):
            widget = gui.CollapsableVert(title, 0.25 * em, gui.Margins(0, 0, 0, 0))
            widget.set_is_open(is_open)
            return widget

        # Model tab: selection, then shape, expression, pose and placement.
        self._shape_section = section("Shape")
        self._expression_section = section("Expression")
        self._pose_settings = section("Pose")
        self._transform_settings = section("Global transform", is_open=False)

        # ----------------------------------- #
        # ------- BODY MODEL SETTINGS ------- #
        # ----------------------------------- #
        self.preload_body_models()
        self._scene.scene.show_ground_plane(self.settings.show_ground, rendering.Scene.GroundPlane(0))

        self._body_model = gui.Combobox()
        for bm in AppWindow.BODY_MODEL_NAMES:
            self._body_model.add_item(bm)

        self._body_model_gender = gui.Combobox()
        for gender in AppWindow.BODY_MODEL_GENDERS[AppWindow.BODY_MODEL_NAMES[0]]:
            self._body_model_gender.add_item(gender)

        # ------- BODY MODEL BETAS SETTINGS ------- #
        self._body_model_shape_comp = gui.Combobox()
        for i in range(AppWindow.BODY_MODEL_N_BETAS[AppWindow.BODY_MODEL_NAMES[0]]):
            self._body_model_shape_comp.add_item(f'{i+1:02d}')

        self._body_beta_val = SliderWithEdit(gui.Slider.DOUBLE, em, decimals=2)
        self._body_beta_val.set_limits(-5.0, 5.0)
        self._body_beta_tensor = torch.zeros(1, 10)
        self._anny_phenotype_values = dict(AppWindow.ANNY_PHENOTYPE_DEFAULTS) #??
        self._body_beta_reset = gui.Button("Reset shape")

        # new global translation gui slider
        self._trans_x = SliderWithEdit(gui.Slider.DOUBLE, em, decimals=3)
        self._trans_x.set_limits(-5.0, 5.0)
        self._trans_y = SliderWithEdit(gui.Slider.DOUBLE, em, decimals=3)
        self._trans_y.set_limits(-5.0, 5.0)
        self._trans_z = SliderWithEdit(gui.Slider.DOUBLE, em, decimals=3)
        self._trans_z.set_limits(-5.0, 5.0)
        self._trans_reset = gui.Button("Reset translation")

        # new global rotation gui slider
        self._rot_x = SliderWithEdit(gui.Slider.DOUBLE, em, decimals=3)
        self._rot_x.set_limits(-5.0, 5.0)
        self._rot_y = SliderWithEdit(gui.Slider.DOUBLE, em, decimals=3)
        self._rot_y.set_limits(-5.0, 5.0)
        self._rot_z = SliderWithEdit(gui.Slider.DOUBLE, em, decimals=3)
        self._rot_z.set_limits(-5.0, 5.0)
        self._rot_reset = gui.Button("Reset rotation")

        # new transparency slider
        self._transparency = SliderWithEdit(gui.Slider.DOUBLE, em, decimals=2)
        self._transparency.set_limits(0.0, 1.0)
        self._transparency.double_value = 0.0
        self._transparency_reset = gui.Button("Reset transparency")

        # camera reset button
        self._camera_reset = gui.Button("Reset camera")

        self._body_beta_text = gui.Label("Betas")
        self._body_beta_text.text = f",".join(f'{x:.1f}'for x in self._body_beta_tensor[0].numpy().tolist())

        # ------- BODY MODEL EXPRESSION SETTINGS ------- #
        self._body_model_exp_comp = gui.Combobox()
        for i in range(10):
            self._body_model_exp_comp.add_item(f'{i + 1:02d}')

        self._body_exp_val = SliderWithEdit(gui.Slider.DOUBLE, em, decimals=2)
        self._body_exp_val.set_limits(-5.0, 5.0)
        self._body_exp_tensor = torch.zeros(1, 10)
        self._body_exp_reset = gui.Button("Reset expression")

        self._body_exp_text = gui.Label("Expression")
        self._body_exp_text.text = f",".join(f'{x:.1f}' for x in self._body_exp_tensor[0].numpy().tolist())


        # ------- BODY MODEL POSE SETTINGS ------- #
        self._body_pose_comp = gui.Combobox()
        for k in AppWindow.POSE_PARAMS[AppWindow.BODY_MODEL_NAMES[0]].keys():
            if k == 'trans':
                continue
            self._body_pose_comp.add_item(k)

        self._body_pose_comp_grid = gui.VGrid(2, 0.25 * em)
        self._body_pose_comp_label = gui.Label("Pose group")
        self._body_pose_joint = gui.Combobox()

        self._body_pose_joint_x = SliderWithEdit(gui.Slider.INT, em)
        self._body_pose_joint_x.set_limits(-180, 180)
        self._body_pose_joint_y = SliderWithEdit(gui.Slider.INT, em)
        self._body_pose_joint_y.set_limits(-180, 180)
        self._body_pose_joint_z = SliderWithEdit(gui.Slider.INT, em)
        self._body_pose_joint_z.set_limits(-180, 180)

        # new slider for MHR
        self._body_pose_joint_val = SliderWithEdit(gui.Slider.DOUBLE, em, decimals=3)
        self._body_pose_joint_val.set_limits(-1.0, 1.0)

        self._body_pose_reset = gui.Button("Reset pose")
        self._body_pose_ik = gui.Button("Run IK")
        self._drag_joints = gui.Checkbox("Drag joints (IK)")
        self._drag_joints.set_on_checked(self._on_drag_joints)

        self._show_joints = gui.Checkbox("Show skeleton")
        self._show_joints.set_on_checked(self._on_show_joints)

        self._show_joint_labels = gui.Checkbox("Show joint labels")
        self._show_joint_labels.set_on_checked(self._on_show_joint_labels)

        # ------- ANNY MODEL SETTINGS ------- #
        anny_settings = gui.CollapsableVert("Anny Settings", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        anny_settings.set_is_open(False)
        # ANNY model type selection
        self._anny_model_type = gui.Combobox()
        for model_type in AppWindow.ANNY_MODEL_TYPES:
            self._anny_model_type.add_item(model_type)
        # ANNY rig selection
        self._anny_rig = gui.Combobox()
        for rig in AppWindow.ANNY_RIGS:
            self._anny_rig.add_item(rig)


        # ANNY phenotype selection
        self._anny_phenotype = gui.Combobox()
        for name in AppWindow.ANNY_PHENOTYPE_NAMES:
            self._anny_phenotype.add_item(name)
        self._anny_phenotype_val = gui.Slider(gui.Slider.DOUBLE)
        self._anny_phenotype_val.set_limits(0.0, 1.0)
        self._anny_reset_shape = gui.Button("Reset Anny shape")

        self._anny_bone_hierarchy = self._build_anny_bone_hierarchy()
        self._anny_visible_bones = []
        self._updating_anny_hierarchy = False
        self._anny_region = gui.Combobox()
        self._anny_group = gui.Combobox()
        self._anny_bone = gui.Combobox()
        self._populate_anny_regions()


        self._on_body_model(AppWindow.BODY_MODEL_NAMES[0], 0)
        self._on_body_pose_comp(list(AppWindow.POSE_PARAMS[AppWindow.BODY_MODEL_NAMES[0]].keys())[0], 0)
        self._body_model.set_on_selection_changed(self._on_body_model)
        self._body_model_gender.set_on_selection_changed(self._on_body_model_gender)

        self._anny_phenotype.set_on_selection_changed(self._on_anny_phenotype_comp)
        self._anny_phenotype_val.set_on_value_changed(self._on_anny_phenotype_val)
        self._anny_reset_shape.set_on_clicked(self._on_anny_reset_shape)
        self._anny_region.set_on_selection_changed(self._on_anny_region)
        self._anny_group.set_on_selection_changed(self._on_anny_group)
        self._anny_bone.set_on_selection_changed(self._on_anny_bone)


        self._body_beta_val.set_on_value_changed(self._on_body_beta_val)
        self._body_beta_reset.set_on_clicked(self._on_body_beta_reset)
        self._body_model_shape_comp.set_on_selection_changed(self._on_body_model_shape_comp)

        self._body_exp_val.set_on_value_changed(self._on_body_exp_val)
        self._body_exp_reset.set_on_clicked(self._on_body_exp_reset)
        self._body_model_exp_comp.set_on_selection_changed(self._on_body_model_exp_comp)

        self._body_pose_comp.set_on_selection_changed(self._on_body_pose_comp)
        self._body_pose_joint.set_on_selection_changed(self._on_body_pose_joint)
        for axis, slider in enumerate((self._body_pose_joint_x, self._body_pose_joint_y,
                                       self._body_pose_joint_z)):
            slider.set_on_value_changed(partial(self._on_body_pose_joint_axis, axis))
        self._body_pose_reset.set_on_clicked(self._on_body_pose_reset)

        self._body_pose_joint_val.set_on_value_changed(self._on_body_pose_joint_val)

        self._body_pose_ik.set_on_clicked(self._on_run_ik)

        # translation callbacks
        for axis, slider in enumerate((self._trans_x, self._trans_y, self._trans_z)):
            slider.set_on_value_changed(partial(self._on_trans, axis))
        self._trans_reset.set_on_clicked(self._on_trans_reset)

        # global rotation callbacks
        for axis, slider in enumerate((self._rot_x, self._rot_y, self._rot_z)):
            slider.set_on_value_changed(partial(self._on_rot, axis))
        self._rot_reset.set_on_clicked(self._on_rot_reset)

        # transparency callbacks
        self._transparency.set_on_value_changed(self._on_transparency)
        self._transparency_reset.set_on_clicked(self._on_transparency_reset)

        # camera reset callback
        self._camera_reset.set_on_clicked(self._on_camera_reset)

        self._scene.set_on_mouse(self._on_mouse_widget)
        self._scene.set_on_key(self._on_key_widget)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Body model"))
        grid.add_child(self._body_model)
        grid.add_child(gui.Label("Gender"))
        grid.add_child(self._body_model_gender)
        self.model_settings.add_child(grid)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Component"))
        grid.add_child(self._body_model_shape_comp)
        grid.add_child(gui.Label("Value"))
        grid.add_child(self._body_beta_val.widget)
        self._shape_section.add_child(grid)

        h = gui.Horiz(0.25 * em)
        h.add_child(self._body_beta_reset)
        self._shape_section.add_child(h)

        # new global translation UI
        h = gui.Horiz(0.25 * em)
        h.add_child(gui.Label("Translation (m)"))
        self._transform_settings.add_child(h)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Translate X"))
        grid.add_child(self._trans_x.widget)
        grid.add_child(gui.Label("Translate Y"))
        grid.add_child(self._trans_y.widget)
        grid.add_child(gui.Label("Translate Z"))
        grid.add_child(self._trans_z.widget)
        self._transform_settings.add_child(grid)
        h = gui.Horiz(0.25 * em)
        h.add_child(self._trans_reset)
        self._transform_settings.add_child(h)

        # new global rotation UI
        self._global_rotation_header_row = gui.Horiz(0.25 * em)
        self._global_rotation_header_row.add_child(gui.Label("Rotation (rad)"))
        self._transform_settings.add_child(self._global_rotation_header_row)
        self._global_rotation_grid = gui.VGrid(2, 0.25 * em)
        self._global_rotation_grid.add_child(gui.Label("Rotate X"))
        self._global_rotation_grid.add_child(self._rot_x.widget)
        self._global_rotation_grid.add_child(gui.Label("Rotate Y"))
        self._global_rotation_grid.add_child(self._rot_y.widget)
        self._global_rotation_grid.add_child(gui.Label("Rotate Z"))
        self._global_rotation_grid.add_child(self._rot_z.widget)
        self._transform_settings.add_child(self._global_rotation_grid)
        self._global_rotation_reset_row = gui.Horiz(0.25 * em)
        self._global_rotation_reset_row.add_child(self._rot_reset)
        self._transform_settings.add_child(self._global_rotation_reset_row)

        self._expression_grid = gui.VGrid(2, 0.25 * em)
        self._expression_grid.add_child(gui.Label("Component"))
        self._expression_grid.add_child(self._body_model_exp_comp)
        self._expression_grid.add_child(gui.Label("Value"))
        self._expression_grid.add_child(self._body_exp_val.widget)
        self._expression_section.add_child(self._expression_grid)

        # Reset expression Button
        self._expression_reset_row = gui.Horiz(0.25 * em)
        self._expression_reset_row.add_child(self._body_exp_reset)
        self._expression_section.add_child(self._expression_reset_row)

        self.model_settings.add_child(self._shape_section)
        self.model_settings.add_child(self._expression_section)
        self.model_settings.add_child(self._pose_settings)
        self.model_settings.add_child(self._transform_settings)

        # Skeleton, labels, drag mode and camera reset live on the viewport
        # toolbar; their checkboxes remain as hidden state holders.

        # transparency slider
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Transparency"))
        grid.add_child(self._transparency.widget)
        self._display_settings.add_child(grid)

        self._display_settings.add_fixed(0.5 * em)

        h = gui.Horiz(0.25 * em)
        h.add_child(self._transparency_reset)
        self._display_settings.add_child(h)

        self._display_settings.add_fixed(em)

        self._body_pose_comp_grid.add_child(self._body_pose_comp_label)
        self._body_pose_comp_grid.add_child(self._body_pose_comp)
        self._pose_settings.add_child(self._body_pose_comp_grid)

        self._joint_grid = gui.VGrid(2, 0.25 * em)
        self._body_pose_joint_label = gui.Label("Joint")
        self._joint_grid.add_child(self._body_pose_joint_label)
        self._joint_grid.add_child(self._body_pose_joint)
        self._pose_settings.add_child(self._joint_grid)

        self._anny_grid = gui.VGrid(2, 0.25 * em)
        self._anny_region_label = gui.Label("Region")
        self._anny_grid.add_child(self._anny_region_label)
        self._anny_grid.add_child(self._anny_region)
        self._anny_group_label = gui.Label("Group")
        self._anny_grid.add_child(self._anny_group_label)
        self._anny_grid.add_child(self._anny_group)
        self._anny_bone_label = gui.Label("Bone")
        self._anny_grid.add_child(self._anny_bone_label)
        self._anny_grid.add_child(self._anny_bone)
        self._pose_settings.add_child(self._anny_grid)

        self._pose_settings.add_fixed(0.5 * em)

        self._rot_grid = gui.VGrid(2, 0.25 * em)
        self._rot_x_label = gui.Label("Rotation X (deg)")
        self._rot_grid.add_child(self._rot_x_label)
        self._rot_grid.add_child(self._body_pose_joint_x.widget)
        self._rot_y_label = gui.Label("Rotation Y (deg)")
        self._rot_grid.add_child(self._rot_y_label)
        self._rot_grid.add_child(self._body_pose_joint_y.widget)
        self._rot_z_label = gui.Label("Rotation Z (deg)")
        self._rot_grid.add_child(self._rot_z_label)
        self._rot_grid.add_child(self._body_pose_joint_z.widget)
        self._pose_settings.add_child(self._rot_grid)

        self._mhr_grid = gui.VGrid(2, 0.25 * em)
        self._mhr_grid.add_child(gui.Label("Value"))
        self._mhr_grid.add_child(self._body_pose_joint_val.widget)
        self._pose_settings.add_child(self._mhr_grid)

        self._pose_settings.add_fixed(0.5 * em)

        h = gui.Horiz(0.25 * em)  # row 2
        h.add_child(self._body_pose_reset)
        self._pose_settings.add_child(h)

        h = gui.Horiz(0.25 * em)  # row 2
        h.add_child(self._body_pose_ik)
        self._pose_settings.add_child(h)

        self._pose_settings.add_child(gui.Label("Toolbar: Drag joints, then drag a marker; release to solve IK."))
        self._display_settings.add_child(self.viewer.view_controls)
        self._display_settings.add_child(self.viewer.material_settings)
        self._display_settings.add_child(self.viewer.advanced_lighting)
        self._set_model_specific_ui_visibility(self._body_model.selected_text)


        # Viewport toolbar: camera presets and the display toggles people reach
        # for most often, floating over the top-left corner of the 3D view.
        self._toolbar = gui.Horiz(
            0.25 * em, gui.Margins(0.5 * em, 0.25 * em, 0.5 * em, 0.25 * em))
        self._toolbar.background_color = gui.Color(0.12, 0.12, 0.12, 0.75)

        def tool_button(text, on_clicked, toggleable=False, tooltip=""):
            button = gui.Button(text)
            button.horizontal_padding_em = 0.5
            button.vertical_padding_em = 0.1
            button.toggleable = toggleable
            button.tooltip = tooltip
            button.set_on_clicked(on_clicked)
            self._toolbar.add_child(button)
            return button

        tool_button("Reset view", self._on_camera_reset, tooltip="Frame the model")
        tool_button("Front", lambda: self._look_from([0., 0., 1.]))
        tool_button("Side", lambda: self._look_from([1., 0., 0.]))
        tool_button("Top", lambda: self._look_from([0., 1., 0.]))
        self._toolbar.add_fixed(0.5 * em)
        self._skeleton_button = tool_button(
            "Skeleton", lambda: self._set_skeleton_visible(self._skeleton_button.is_on),
            toggleable=True, tooltip="Show joints and bones")
        self._labels_button = tool_button(
            "Labels", lambda: self._set_joint_labels_visible(self._labels_button.is_on),
            toggleable=True, tooltip="Show joint names")
        self._drag_button = tool_button(
            "Drag joints", lambda: self._set_drag_joints(self._drag_button.is_on),
            toggleable=True, tooltip="Drag a joint marker to pose with IK")
        self._toolbar_state = [
            (self._skeleton_button, self._show_joints),
            (self._labels_button, self._show_joint_labels),
            (self._drag_button, self._drag_joints),
        ]

        # Info panel
        self.info = gui.Label("")
        self.info.visible = False

        # Reserve a separate viewport and full-height tabbed sidebar.
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._toolbar)
        w.add_child(self._settings_panel)
        w.add_child(self.info)
        # w.add_child(self.joint_label_3d)

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("Help", AppWindow.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", AppWindow.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item("Open", AppWindow.MENU_OPEN)
            file_menu.add_item("Export Current Image", AppWindow.MENU_EXPORT)
            file_menu.add_item("Save Model Params", AppWindow.MENU_SAVE)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.add_item("Model - Lighting - Materials",
                                   AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT,
                                     self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_SAVE,
                                     self._on_save_dialog)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

        self.viewer.apply()

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        em = layout_context.theme.font_size
        width = min(26 * em, int(r.width * 0.45))
        sidebar_width = width if self._settings_panel.visible else 0
        self._scene.frame = gui.Rect(r.x, r.y, r.width - sidebar_width, r.height)
        self._settings_panel.frame = gui.Rect(
            r.get_right() - width, r.y, width, r.height)

        pref = self._toolbar.calc_preferred_size(layout_context,
                                                 gui.Widget.Constraints())
        margin = int(0.5 * em)
        self._toolbar.frame = gui.Rect(
            self._scene.frame.x + margin, self._scene.frame.y + margin,
            min(pref.width, max(self._scene.frame.width - 2 * margin, 0)), pref.height)

        pref = self.info.calc_preferred_size(layout_context,
                                             gui.Widget.Constraints())
        self.info.frame = gui.Rect(r.x,
                                   r.get_bottom() - pref.height, pref.width,
                                   pref.height)

    def _active_model(self):
        bm = self._body_model.selected_text
        gender = self._body_model_gender.selected_text
        return AppWindow.PRELOADED_BODY_MODELS[f'{bm.lower()}-{gender.lower()}']

    def _current_joint_display_names(self):
        joints = AppWindow.JOINTS
        joint_count = 0 if joints is None else joints.shape[0]
        joint_names = list(self._active_model().joint_names)
        if len(joint_names) < joint_count:
            joint_names.extend(str(i) for i in range(len(joint_names), joint_count))
        return joint_names

    def _on_show_joint_labels(self, show):
        for label3d in self.joint_labels_3d_list:
            self._scene.remove_3d_label(label3d)
        self.joint_labels_3d_list.clear()
        if show and AppWindow.JOINTS is not None:
            joint_names = self._current_joint_display_names()
            for position, name in zip(AppWindow.JOINTS, joint_names):
                self.joint_labels_3d_list.append(
                    self._scene.add_3d_label(position, name)
                )
        self._sync_toolbar()

    def _skeleton_edges(self, num_joints):
        """Use the active model's kinematic tree, excluding extra landmarks."""
        return np.asarray([
            [int(parent), child]
            for child, parent in enumerate(self._active_model().joint_parents)
            if child < num_joints and 0 <= int(parent) < num_joints
            and int(parent) != child
        ], dtype=np.int32).reshape(-1, 2)

    def _on_show_joints(self, show, update_transparency=True):
        bm = self._body_model.selected_text
        if update_transparency and self._scene.scene.has_geometry("__body_model__"):
            if show:
                suggested = 0.2
                AppWindow.TRANSPARENCY[bm] = suggested
                self._transparency.double_value = suggested
                self._apply_transparency(suggested)
            else:

                AppWindow.TRANSPARENCY[bm] = 0.0
                self._transparency.double_value = 0.0
                self._apply_transparency(0.0)

        joints = AppWindow.JOINTS
        num_joints = joints.shape[0] if joints is not None else 0

        if self._scene.scene.has_geometry("__skeleton__"):
            self._scene.scene.remove_geometry("__skeleton__")
        if show and num_joints:
            edges = self._skeleton_edges(num_joints)
            if len(edges):
                skeleton = o3d.geometry.LineSet(
                    o3d.utility.Vector3dVector(joints),
                    o3d.utility.Vector2iVector(edges),
                )
                skeleton.paint_uniform_color([0.7, 0.3, 0.3])
                bone_material = rendering.MaterialRecord()
                bone_material.shader = "unlitLine"
                bone_material.line_width = 4.0
                self._scene.scene.add_geometry(
                    "__skeleton__", skeleton, bone_material
                )


        # Keep spheres between updates; remove only joints no longer displayed.
        for i in list(self._joint_geometry_radii):
            if not show or i >= num_joints:
                self._scene.scene.remove_geometry(f"__joints_{i}__")
                del self._joint_geometry_radii[i]

        green = [0.3, 0.7, 0.3, 1.0]
        red = [0.7, 0.3, 0.3, 1.0]
        hand_radius = foot_radius = 0.01
        head_radius = 0.004
        body_radius = 0.025
        joint_names = self._current_joint_display_names()

        mat = rendering.MaterialRecord()
        mat.base_color = red
        mat.shader = "defaultLit"

        mat_selected = rendering.MaterialRecord()
        mat_selected.base_color = green
        mat_selected.shader = "defaultLit"

        joints = AppWindow.JOINTS
        if show:
            for i in range(num_joints):
                joint_name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
                if bm == "ANNY":
                    if joint_name in AppWindow.ANNY_HAND_JOINT_NAMES:
                        radius = hand_radius
                    elif joint_name in AppWindow.ANNY_FOOT_JOINT_NAMES:
                        radius = foot_radius
                    elif joint_name in AppWindow.ANNY_HEAD_JOINT_NAMES:
                        radius = head_radius
                    elif joint_name in AppWindow.ANNY_BODY_JOINT_NAMES:
                        radius = body_radius
                    else:
                        logger.warning(
                            f'Unknown ANNY joint "{joint_name}"; using body radius'
                        )
                        radius = body_radius
                elif bm == "MHR":
                    if joint_name in AppWindow.MHR_HAND_JOINT_NAMES:
                        radius = hand_radius
                    elif joint_name in AppWindow.MHR_FOOT_JOINT_NAMES:
                        radius = foot_radius
                    elif joint_name in AppWindow.MHR_HEAD_JOINT_NAMES:
                        radius = head_radius
                    else:
                        radius = body_radius
                else:
                    radius = body_radius
                    if joint_name in LEFT_HAND_KEYPOINT_NAMES + RIGHT_HAND_KEYPOINT_NAMES:
                        radius = hand_radius
                    elif joint_name in HEAD_KEYPOINT_NAMES:
                        radius = head_radius
                    elif joint_name in FOOT_KEYPOINT_NAMES:
                        radius = foot_radius

                current_mat = mat_selected if (AppWindow.SELECTED_JOINT is not None) and (i == AppWindow.SELECTED_JOINT)\
                    else mat

                transform = np.eye(4)
                transform[:3, 3] = joints[i]

                geometry_name = f"__joints_{i}__"
                if self._joint_geometry_radii.get(i) != radius:
                    if i in self._joint_geometry_radii:
                        self._scene.scene.remove_geometry(geometry_name)
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                    sphere.compute_vertex_normals()
                    self._scene.scene.add_geometry(geometry_name, sphere, current_mat)
                    self._joint_geometry_radii[i] = radius
                else:
                    self._scene.scene.modify_geometry_material(geometry_name, current_mat)
                # Spheres remain centered at their local origin; the transform
                # supplies the entire world position on every update.
                self._scene.scene.set_geometry_transform(geometry_name, transform)

        self._on_show_joint_labels(self._show_joint_labels.checked)
      
    def _set_model_specific_ui_visibility(self, body_model_name):
        is_mhr = body_model_name == "MHR"
        is_flame = body_model_name == "FLAME"
        self._body_pose_joint_x.visible = not is_mhr
        self._body_pose_joint_y.visible = not is_mhr
        self._body_pose_joint_z.visible = not is_mhr
        self._body_pose_joint_val.visible = is_mhr
       # Global rotation sliders are visible for every model.
        if hasattr(self, "_global_rotation_header_row"):
            self._global_rotation_header_row.visible = True
        if hasattr(self, "_global_rotation_grid"):
            self._global_rotation_grid.visible = True
        if hasattr(self, "_global_rotation_reset_row"):
            self._global_rotation_reset_row.visible = True
        # MHR edits scalar parameters, so hide the per-axis rotation grid.
        if hasattr(self, "_rot_grid"):
            self._rot_grid.visible = not is_mhr
        if hasattr(self, '_mhr_grid'):
            self._mhr_grid.visible = is_mhr
        self.window.set_needs_layout()
        is_anny = body_model_name == "ANNY"
        # Only SMPL-X, FLAME and MHR have facial expressions.
        has_expression = body_model_name in ("SMPLX", "FLAME", "MHR")
        if hasattr(self, "_expression_section"):
            self._expression_section.visible = has_expression
        if hasattr(self, "_expression_grid"):
            self._expression_grid.visible = has_expression
        if hasattr(self, "_expression_reset_row"):
            self._expression_reset_row.visible = has_expression
        if hasattr(self, "_body_pose_joint_label"):
            self._joint_grid.visible = not is_anny
        if hasattr(self, "_anny_grid"):
            self._anny_grid.visible = is_anny
        if hasattr(self, '_joint_grid'):
            self._joint_grid.visible = not is_flame
        self.window.set_needs_layout()

    def _classify_anny_bone(self, bone_name):
        if bone_name.endswith(".L"):
            side = "Left"
            base_name = bone_name[:-2]
        elif bone_name.endswith(".R"):
            side = "Right"
            base_name = bone_name[:-2]
        else:
            side = None
            base_name = bone_name

        if base_name == "root":
            return "Root", "Root"

        leg_groups = {
            "pelvis": "Pelvis",
            "upperleg": "Upper Leg",
            "lowerleg": "Lower Leg",
            "foot": "Foot",
            "toe": "Toes",
        }
        for prefix, group in leg_groups.items():
            if base_name.startswith(prefix):
                region = f"{side} Leg" if side else "Torso"
                return region, group

        arm_groups = {
            "clavicle": "Clavicle",
            "shoulder": "Shoulder",
            "upperarm": "Upper Arm",
            "lowerarm": "Lower Arm",
            "wrist": "Wrist",
            "finger": "Fingers",
            "metacarpal": "Fingers",
        }
        for prefix, group in arm_groups.items():
            if base_name.startswith(prefix):
                region = f"{side} Arm" if side else "Other"
                return region, group

        if base_name.startswith("spine"):
            return "Torso", "Spine"
        if base_name.startswith("breast"):
            return "Torso", "Breast"
        if base_name.startswith("neck"):
            return "Head & Face", "Neck"
        if base_name == "head":
            return "Head & Face", "Head"
        if base_name == "jaw":
            return "Head & Face", "Jaw"
        if base_name.startswith("tongue"):
            return "Head & Face", "Tongue"
        if base_name.startswith(("eye", "oculi", "orbicularis")):
            return "Head & Face", "Eyes"
        if base_name.startswith(("oris", "risorius")):
            return "Head & Face", "Mouth"
        if base_name.startswith(("levator", "temporalis")):
            return "Head & Face", "Face Muscles"
        if base_name.startswith("special"):
            return "Head & Face", "Other Face Controls"

        return "Other", "Other"

    def _build_anny_bone_hierarchy(self):
        hierarchy = {region: {} for region in AppWindow.ANNY_REGION_ORDER}
        # ANNY may be absent (not installed, or excluded from a trimmed build).
        bone_names = AppWindow.JOINT_NAMES.get("ANNY", {}).get("pose", [])
        for bone_index, bone_name in enumerate(bone_names):
            # Hide the root (index 0); it is driven by the global rotation sliders.
            if bone_index == 0:
                continue
            region, group = self._classify_anny_bone(bone_name)
            hierarchy.setdefault(region, {})
            hierarchy[region].setdefault(group, [])
            hierarchy[region][group].append((bone_index, bone_name))
        return {
            region: groups
            for region, groups in hierarchy.items()
            if groups
        }

    def _populate_anny_regions(self):
        self._updating_anny_hierarchy = True
        self._anny_region.clear_items()
        for region in self._anny_bone_hierarchy.keys():
            self._anny_region.add_item(region)
        if self._anny_region.number_of_items > 0:
            self._anny_region.selected_index = 0
        self._updating_anny_hierarchy = False
        self._populate_anny_groups(select_first=False)

    def _populate_anny_groups(self, select_first=True):
        region = self._anny_region.selected_text
        groups = self._anny_bone_hierarchy.get(region, {})

        self._updating_anny_hierarchy = True
        self._anny_group.clear_items()
        for group in groups.keys():
            self._anny_group.add_item(group)
        if self._anny_group.number_of_items > 0:
            self._anny_group.selected_index = 0
        self._updating_anny_hierarchy = False
        self._populate_anny_bones(select_first=select_first)

    def _populate_anny_bones(self, select_first=True):
        region = self._anny_region.selected_text
        group = self._anny_group.selected_text
        self._anny_visible_bones = self._anny_bone_hierarchy.get(region, {}).get(group, [])

        self._updating_anny_hierarchy = True
        self._anny_bone.clear_items()
        for bone_index, bone_name in self._anny_visible_bones:
            self._anny_bone.add_item(f"{bone_index}-{bone_name}")
        if self._anny_bone.number_of_items > 0:
            self._anny_bone.selected_index = 0
        self._updating_anny_hierarchy = False

        if select_first and self._anny_visible_bones:
            self._select_anny_bone_index(self._anny_visible_bones[0][0])

    def _select_anny_bone_index(self, bone_index):
        if self._body_model.selected_text != "ANNY":
            return
        if self._body_pose_comp.selected_text != "pose":
            self._body_pose_comp.selected_index = 0
            self._on_body_pose_comp("pose", 0)
        if bone_index < self._body_pose_joint.number_of_items:
            self._updating_anny_hierarchy = True
            self._body_pose_joint.selected_index = bone_index
            self._updating_anny_hierarchy = False
            self._sync_pose_joint_sliders()

    def _sync_anny_hierarchy_to_bone(self, bone_index):
        if self._body_model.selected_text != "ANNY":
            return

        bone_name = AppWindow.JOINT_NAMES["ANNY"]["pose"][bone_index]
        region, group = self._classify_anny_bone(bone_name)
        if region not in self._anny_bone_hierarchy:
            return

        self._updating_anny_hierarchy = True
        region_names = list(self._anny_bone_hierarchy.keys())
        self._anny_region.selected_index = region_names.index(region)

        groups = self._anny_bone_hierarchy[region]
        group_names = list(groups.keys())
        self._anny_group.clear_items()
        for group_name in group_names:
            self._anny_group.add_item(group_name)
        self._anny_group.selected_index = group_names.index(group)

        self._anny_visible_bones = groups[group]
        self._anny_bone.clear_items()
        selected_bone_index = 0
        for visible_index, (candidate_index, candidate_name) in enumerate(self._anny_visible_bones):
            self._anny_bone.add_item(f"{candidate_index}-{candidate_name}")
            if candidate_index == bone_index:
                selected_bone_index = visible_index
        self._anny_bone.selected_index = selected_bone_index
        self._updating_anny_hierarchy = False

    def _on_anny_region(self, name, index):
        if self._updating_anny_hierarchy:
            return
        self._populate_anny_groups(select_first=True)

    def _on_anny_group(self, name, index):
        if self._updating_anny_hierarchy:
            return
        self._populate_anny_bones(select_first=True)

    def _on_anny_bone(self, name, index):
        if self._updating_anny_hierarchy:
            return
        if index < len(self._anny_visible_bones):
            self._select_anny_bone_index(self._anny_visible_bones[index][0])

    def _on_body_model(self, name, index):
        logger.info(f"Loading body model {name}-{index}")
        # not resetting expression anymore instead restore old values
        self._body_exp_tensor = self._get_or_create_exp_tensor(name)
        self._body_exp_val.double_value = self._body_exp_tensor[0, 0].item()
        AppWindow.CAM_FIRST = True
        self._set_model_specific_ui_visibility(name)

        # treat anny phenotypes like smpl betas
        self._body_model_shape_comp.clear_items()

        if name == "ANNY":
            for phenotype in AppWindow.ANNY_PHENOTYPE_NAMES:
                self._body_model_shape_comp.add_item(phenotype)
            self._body_beta_val.set_limits(0.0, 1.0)
            self._body_beta_val.double_value = self._anny_phenotype_values[AppWindow.ANNY_PHENOTYPE_NAMES[0]]

        else:
            for i in range(AppWindow.BODY_MODEL_N_BETAS[name]):
                self._body_model_shape_comp.add_item(f"{i+1:02d}")
            self._body_beta_val.set_limits(-5.0, 5.0)
            # restore betas not reset
            self._body_beta_tensor = self._get_or_create_beta_tensor(name)
            self._body_beta_val.double_value = self._body_beta_tensor[0, 0].item()
            self._body_beta_text.text = f",".join(f'{x:.1f}' for x in self._body_beta_tensor[0].numpy().tolist())

        self._body_model_gender.clear_items()
        for gender in AppWindow.BODY_MODEL_GENDERS[name]:
            self._body_model_gender.add_item(gender)

        AppWindow.GROUND_OFFSET = None
        self.load_body_model(name, gender=self._body_model_gender.selected_text)

        self._body_pose_comp.clear_items()
        for k in AppWindow.POSE_PARAMS[name].keys():
            # trans and global_orient are driven by the global sliders.
            if k in ('trans', 'global_orient'):
                continue
            self._body_pose_comp.add_item(k)

        self._body_pose_joint.clear_items()
        joint_names = AppWindow.JOINT_NAMES[name][self._body_pose_comp.selected_text]
        for i in range(AppWindow.POSE_PARAMS[name][self._body_pose_comp.selected_text].shape[1]):
            # MHR: hide the first 6 (root translation + rotation), which have their own sliders.
            if name == 'MHR' and i < 6:
                continue
            # MHR: hide parameters with min == max (not movable).
            if name == 'MHR':
                lo, hi = self._mhr_limits(i)
                if lo == hi:
                    continue
            # SUPR/STAR/ANNY: hide the root (index 0); it has the global rotation sliders.
            if name in ('SUPR', 'STAR', 'ANNY') and i == 0:
                continue
            self._body_pose_joint.add_item(f'{i}-{joint_names[i]}')
        if name == "ANNY" and self._body_pose_joint.number_of_items > 0:
            self._sync_anny_hierarchy_to_bone(0)

        self._sync_pose_joint_sliders()
        self._sync_global_sliders()
        self._sync_transparency()
        AppWindow.SELECTED_JOINT = None
        self._on_show_joints(self._show_joints.checked, update_transparency=False)

        # Only SMPL-X and FLAME have more than one pose group.
        need_comp = name in ('SMPLX', 'FLAME')
        self._body_pose_comp_grid.visible = need_comp

        self._body_pose_comp_label.text = 'Pose group'

    def _on_body_model_gender(self, name, index):
        logger.info(f"Changing {self._body_model.selected_text} body model gender to {name}-{index}")
        self._body_beta_val.double_value = 0.0
        AppWindow.GROUND_OFFSET = None
        self.load_body_model(self._body_model.selected_text, gender=name)
        self._sync_beta_slider()
        self._sync_pose_joint_sliders()
        self._sync_global_sliders()
        self._sync_transparency()
        self._on_show_joints(self._show_joints.checked, update_transparency=False)

    def _on_body_beta_val(self, val):
        if self._body_model.selected_text == 'ANNY':
            name = self._body_model_shape_comp.selected_text
            self._anny_phenotype_values[name] = float(val)
        else:
            self._body_beta_tensor[0, int(self._body_model_shape_comp.selected_text)-1] = float(val)
            self._body_beta_text.text = f",".join(f'{x:.1f}' for x in self._body_beta_tensor[0].numpy().tolist())
        AppWindow.GROUND_OFFSET = None
        self.load_body_model()

    def _on_body_exp_val(self, val):
        self._body_exp_tensor[0, int(self._body_model_exp_comp.selected_text)-1] = float(val)
        self._body_exp_text.text = f",".join(f'{x:.1f}' for x in self._body_exp_tensor[0].numpy().tolist())
        self.load_body_model()

    def _mhr_limits(self, index):
        """(min, max) of one MHR model parameter for the selected MHR variant."""
        gender = self._body_model_gender.selected_text
        wrapper = AppWindow.PRELOADED_BODY_MODELS[f'mhr-{gender.lower()}']
        return wrapper.parameter_limits['model_parameters'][index]

    def _sync_mhr_value_slider(self, index):
        lo, hi = self._mhr_limits(index)
        self._body_pose_joint_val.set_limits(float(lo), float(hi))
        self._body_pose_joint_val.double_value = float(
            AppWindow.POSE_PARAMS['MHR']['model_parameters'][0, index])

    def _on_body_pose_joint(self, name, index):
        self._sync_pose_joint_sliders()
        if self._body_model.selected_text == "ANNY" and not self._updating_anny_hierarchy:
            bone_index = int(name.split('-')[0])
            self._sync_anny_hierarchy_to_bone(bone_index)
        elif self._body_model.selected_text == "MHR":
            self._sync_mhr_value_slider(int(name.split('-')[0]))

    def _on_body_pose_joint_axis(self, axis, val):
        """Per-joint Euler rotation slider for one axis, in degrees."""
        bm = self._body_model.selected_text
        bp = self._body_pose_comp.selected_text
        ji = int(self._body_pose_joint.selected_text.split('-')[0])
        euler_angle = [self._body_pose_joint_x.int_value, self._body_pose_joint_y.int_value,
                       self._body_pose_joint_z.int_value]
        euler_angle[axis] = val
        axis_angle = euler_to_rotvec_for_body_model(bm, euler_angle)
        AppWindow.POSE_PARAMS[bm][bp][0, ji] = torch.from_numpy(axis_angle).float()
        self.load_body_model()

    def _on_body_pose_joint_val(self, val):
        bm = self._body_model.selected_text
        bp = self._body_pose_comp.selected_text
        ji = int(self._body_pose_joint.selected_text.split('-')[0])
        AppWindow.POSE_PARAMS[bm][bp][0, ji] = float(val)

        # MHR: Root-Translation/Rotation Sync mit trans/global_orient
        if bm == 'MHR':
            if ji in [0, 1, 2]:
                AppWindow.POSE_PARAMS['MHR']['trans'][0, 0, ji] = float(val)
            elif ji in [3, 4, 5]:
                AppWindow.POSE_PARAMS['MHR']['global_orient'][0, 0, ji - 3] = float(val)

        self.load_body_model()

    def _on_body_model_shape_comp(self, name, index):
        if self._body_model.selected_text == 'ANNY':
            self._body_beta_val.double_value = self._anny_phenotype_values.get(name, 0.5)
        else:
            self._body_beta_val.double_value = self._body_beta_tensor[0, index].item()

    def _on_body_model_exp_comp(self, name, index):
        self._body_exp_val.double_value = self._body_exp_tensor[0, index].item()

    def _on_body_pose_comp(self, name, index):
        if self._body_model.selected_text == "MHR":
            self._sync_mhr_value_slider(int(name.split('-')[0]))
        self._body_pose_joint.clear_items()
        bm = self._body_model.selected_text
        joint_names = AppWindow.JOINT_NAMES[bm][name]
        for i in range(AppWindow.POSE_PARAMS[bm][name].shape[1]):
            # MHR: hide the first 6 (root translation + rotation), which have their own sliders.
            if bm == 'MHR' and i < 6:
                continue
            # MHR: hide parameters with min == max (not movable).
            if bm == 'MHR':
                lo, hi = self._mhr_limits(i)
                if lo == hi:
                    continue
            # SUPR/STAR/ANNY: hide the root (index 0).
            if bm in ('SUPR', 'STAR', 'ANNY') and i == 0:
                continue
            self._body_pose_joint.add_item(f'{i}-{joint_names[i]}')
        self._sync_pose_joint_sliders()


    def _on_body_beta_reset(self):
        if self._body_model.selected_text == 'ANNY':
            self._anny_phenotype_values = dict(AppWindow.ANNY_PHENOTYPE_DEFAULTS)
            self._body_beta_val.double_value = self._anny_phenotype_values[
            self._body_model_shape_comp.selected_text
        ]
        else:
            self._body_beta_tensor = torch.zeros(1, AppWindow.BODY_MODEL_N_BETAS[self._body_model.selected_text])
            AppWindow.BODY_BETAS[self._body_model.selected_text] = self._body_beta_tensor
            self._body_beta_text.text = f",".join(f'{x:.1f}' for x in self._body_beta_tensor[0].numpy().tolist())
            self._body_beta_val.double_value = 0.0
        AppWindow.GROUND_OFFSET = None
        self.load_body_model()

    def _on_anny_phenotype_comp(self, name, index):
        self._anny_phenotype_val.double_value = self._anny_phenotype_values[name]

    def _on_anny_phenotype_val(self, val):
        name = self._anny_phenotype.selected_text
        self._anny_phenotype_values[name] = float(val)
        if self._body_model.selected_text == 'ANNY':
            AppWindow.GROUND_OFFSET = None
            self.load_body_model()

    def _on_anny_reset_shape(self):
        self._anny_phenotype_values = dict(AppWindow.ANNY_PHENOTYPE_DEFAULTS)
        self._anny_phenotype_val.double_value = self._anny_phenotype_values[
            self._anny_phenotype.selected_text
        ]
        if self._body_model.selected_text == 'ANNY':
            AppWindow.GROUND_OFFSET = None
            self.load_body_model()

    def _on_body_exp_reset(self):
        self._body_exp_tensor = torch.zeros(1, 10)
        AppWindow.BODY_EXPS[self._body_model.selected_text] = self._body_exp_tensor
        self._body_exp_text.text = f",".join(f'{x:.1f}' for x in self._body_exp_tensor[0].numpy().tolist())
        self._body_exp_val.double_value = 0.0
        self.load_body_model()

    def _on_body_pose_reset(self):
        bm = self._body_model.selected_text
        for key in AppWindow.POSE_PARAMS[bm]:
            AppWindow.POSE_PARAMS[bm][key] = torch.zeros_like(AppWindow.POSE_PARAMS[bm][key])
        self._sync_pose_joint_sliders()
        self.load_body_model()

    # NEW GLOBAL TRANSLATION
    def _on_trans(self, axis, val):
        """Global translation slider for one axis (0=X, 1=Y, 2=Z)."""
        params = AppWindow.POSE_PARAMS[self._body_model.selected_text]
        if "trans" not in params:
            return
        params["trans"][0, 0, axis] = val
        self.load_body_model()

    def _on_trans_reset(self):
        params = AppWindow.POSE_PARAMS[self._body_model.selected_text]
        if "trans" not in params:
            return
        params["trans"] = torch.zeros_like(params["trans"])
        for slider in (self._trans_x, self._trans_y, self._trans_z):
            slider.double_value = 0.0
        self.load_body_model()

    # ANNY's root rotation uses another axis order: viewer X, Y, Z map to
    # pose components 0, 2 and -1.
    ANNY_ROOT_AXES = ((0, 1.0), (2, 1.0), (1, -1.0))

    def _set_global_rotation(self, axis, val):
        """Write one global rotation component; False if the model has none."""
        bm = self._body_model.selected_text
        params = AppWindow.POSE_PARAMS[bm]
        if "global_orient" in params:
            params["global_orient"][0, 0, axis] = val
            if bm == 'MHR':
                params['model_parameters'][0, 3 + axis] = val
        elif "pose" in params:
            if bm == 'ANNY':
                index, sign = self.ANNY_ROOT_AXES[axis]
                params["pose"][0, 0, index] = sign * val
            else:
                params["pose"][0, 0, axis] = val
        else:
            return False
        return True

    def _on_rot(self, axis, val):
        """Global rotation slider for one axis, in radians."""
        if self._set_global_rotation(axis, val):
            self.load_body_model()

    def _on_rot_reset(self):
        if not all([self._set_global_rotation(axis, 0.0) for axis in range(3)]):
            return
        for slider in (self._rot_x, self._rot_y, self._rot_z):
            slider.double_value = 0.0
        self.load_body_model()

    def _on_transparency(self, val):
        AppWindow.TRANSPARENCY[self._body_model.selected_text] = val
        self._apply_transparency(val)

    def _on_transparency_reset(self):
        AppWindow.TRANSPARENCY[self._body_model.selected_text] = 0.0
        if self._scene.scene.has_geometry("__body_model__"):
            mat_body = rendering.MaterialRecord()
            mat_body.shader = "defaultLit"
            mat_body.base_color = [0.9, 0.9, 0.9, 1.0]
            self._scene.scene.modify_geometry_material("__body_model__", mat_body)
        self._transparency.double_value = 0.0

    def _on_camera_reset(self):
        if AppWindow.CAM_BOUNDS is not None:
            self._scene.setup_camera(60, AppWindow.CAM_BOUNDS, AppWindow.CAM_CENTER)

    def _look_from(self, direction):
        """Place the camera on the given side of the model, looking at its center."""
        if AppWindow.CAM_BOUNDS is None:
            return
        center = np.asarray(AppWindow.CAM_CENTER, dtype=np.float32)
        direction = np.asarray(direction, dtype=np.float32)
        distance = 1.8 * float(np.max(AppWindow.CAM_BOUNDS.get_extent()))
        eye = center + direction * distance
        up = np.array([0., 0., -1.], dtype=np.float32) if abs(direction[1]) > 0.9 \
            else np.array([0., 1., 0.], dtype=np.float32)
        self._scene.look_at(center, eye, up)

    def _set_skeleton_visible(self, show):
        self._show_joints.checked = show
        self._on_show_joints(show)

    def _set_joint_labels_visible(self, show):
        self._show_joint_labels.checked = show
        self._on_show_joint_labels(show)

    def _set_drag_joints(self, enabled):
        self._drag_joints.checked = enabled
        self._on_drag_joints(enabled)

    def _sync_toolbar(self):
        """Mirror the sidebar checkboxes on the toolbar toggles."""
        for button, checkbox in getattr(self, '_toolbar_state', ()):
            button.is_on = checkbox.checked

    def _on_drag_joints(self, enabled):
        if AppWindow.IS_DRAGGING:
            self._finish_joint_drag()
        AppWindow.DRAG_KEY_PRESSED = False
        if enabled:
            self._show_joints.checked = True
            self._on_show_joints(True)
        self._update_label("Drag a joint marker; release to solve IK." if enabled
                           else "Camera controls restored.")
        self._sync_toolbar()

    def _finish_joint_drag(self):
        AppWindow.IS_DRAGGING = False
        AppWindow.DRAG_DEPTH = None
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        if self._ik_target_dirty:
            self._on_run_ik()

    def _on_key_widget(self, event):
        if event.key == gui.KeyName.Q:
            AppWindow.DRAG_KEY_PRESSED = event.type == gui.KeyEvent.Type.DOWN
            if not AppWindow.DRAG_KEY_PRESSED and AppWindow.IS_DRAGGING:
                self._finish_joint_drag()
            return gui.Widget.EventCallbackResult.HANDLED

        offsets = {
            gui.KeyName.ONE: [-0.01, 0, 0], gui.KeyName.TWO: [0.01, 0, 0],
            gui.KeyName.THREE: [0, -0.01, 0], gui.KeyName.FOUR: [0, 0.01, 0],
            gui.KeyName.FIVE: [0, 0, -0.01], gui.KeyName.SIX: [0, 0, 0.01],
        }
        selected = AppWindow.SELECTED_JOINT
        if (event.type == gui.KeyEvent.Type.DOWN and self._show_joints.checked
                and selected is not None and AppWindow.JOINTS is not None
                and selected < len(AppWindow.JOINTS) and event.key in offsets):
            AppWindow.JOINTS[selected] += np.asarray(offsets[event.key])
            self._ik_target_dirty = True
            self._on_show_joints(True, update_transparency=False)
            self._update_label('Target moved. Click Run IK to fit the pose.')
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _project_joint_positions(self):
        """Return viewport pixels and depth for picking the visible joint markers."""
        joints = AppWindow.JOINTS
        camera = self._scene.scene.camera
        positions = np.column_stack((joints, np.ones(len(joints))))
        clip = positions @ np.asarray(camera.get_view_matrix()).T
        clip = clip @ np.asarray(camera.get_projection_matrix()).T
        valid = clip[:, 3] > 1e-8
        ndc = clip[:, :3] / np.where(valid, clip[:, 3], 1.)[:, None]
        frame = self._scene.frame
        pixels = np.column_stack(((ndc[:, 0] + 1) * frame.width / 2,
                                  (1 - ndc[:, 1]) * frame.height / 2))
        depth = (ndc[:, 2] + 1) / 2
        valid &= np.isfinite(pixels).all(axis=1) & (depth >= 0) & (depth < 1)
        return pixels, depth, valid

    def _on_mouse_widget(self, event):
        frame = self._scene.frame
        x, y = event.x - frame.x, event.y - frame.y
        if (event.type == gui.MouseEvent.Type.BUTTON_DOWN
                and (AppWindow.DRAG_KEY_PRESSED or self._drag_joints.checked)
                and self._show_joints.checked
                and AppWindow.JOINTS is not None and len(AppWindow.JOINTS)):
            if not (0 <= x < frame.width and 0 <= y < frame.height):
                return gui.Widget.EventCallbackResult.IGNORED
            pixels, depths, valid = self._project_joint_positions()
            distances = np.sum((pixels - [x, y]) ** 2, axis=1)
            distances[~valid] = np.inf
            selected = int(np.argmin(distances))
            if distances[selected] > max(12., self.em) ** 2:
                self._update_label('Click a joint marker to select it.')
                return gui.Widget.EventCallbackResult.HANDLED
            AppWindow.SELECTED_JOINT = selected
            AppWindow.IS_DRAGGING = True
            AppWindow.DRAG_DEPTH = float(depths[selected])
            self._scene.set_view_controls(gui.SceneWidget.Controls.PICK_POINTS)
            name = self._current_joint_display_names()[selected]
            self._update_label(f'Selected {name}. Drag the marker; release to solve IK.')
            self._on_show_joints(True, update_transparency=False)
            return gui.Widget.EventCallbackResult.HANDLED

        if (event.type in (gui.MouseEvent.Type.DRAG, gui.MouseEvent.Type.MOVE)
                and AppWindow.IS_DRAGGING
                and AppWindow.DRAG_DEPTH is not None):
            world = self._scene.scene.camera.unproject(
                x, y, AppWindow.DRAG_DEPTH, frame.width, frame.height)
            AppWindow.JOINTS[AppWindow.SELECTED_JOINT] = np.asarray(world[:3])
            self._ik_target_dirty = True
            self._on_show_joints(True, update_transparency=False)
            return gui.Widget.EventCallbackResult.HANDLED

        if event.type == gui.MouseEvent.Type.BUTTON_UP and AppWindow.IS_DRAGGING:
            self._finish_joint_drag()
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED


    def _update_label(self, text):
        self.info.text = text
        self.info.visible = (text != "")
        # We are sizing the info label to be exactly the right size,
        # so since the text likely changed width, we need to
        # re-layout to set the new frame.
        self.window.set_needs_layout()

    def _on_run_ik(self):
        if AppWindow.JOINTS is None:
            return
        bm = self._body_model.selected_text
        # Model adapters already include user translation and root rotation.
        # Only the viewer's ground offset needs to be removed from the targets.
        targets = AppWindow.JOINTS - np.array([0., AppWindow.GROUND_OFFSET or 0., 0.])
        try:
            result = solve_ik(
                self._active_model(), AppWindow.POSE_PARAMS[bm], targets,
                self.get_shape_params(), self._body_exp_tensor,
            )
        except (ValueError, RuntimeError) as error:
            logger.exception(f'IK failed for {bm}: {error}')
            self._update_label(f'IK could not solve this target: {error}')
            return
        AppWindow.POSE_PARAMS[bm] = result.parameters
        self.load_body_model()
        self._sync_pose_joint_sliders()
        self._sync_global_sliders()
        if result.initial_error <= 0.0005:
            self._update_label('Enable Drag joints in Pose and drag a marker, or hold Q while dragging.')
        elif result.final_error >= result.initial_error * 0.999:
            self._update_label('Target cannot be reached with the current articulation. Global placement is fixed.')
        else:
            self._update_label(
                f'IK target error: {result.initial_error * 1000:.1f} -> '
                f'{result.final_error * 1000:.1f} mm')
        return result

    def _reset_rot_sliders(self):
        self._body_pose_joint_x.int_value = 0
        self._body_pose_joint_y.int_value = 0
        self._body_pose_joint_z.int_value = 0
        self._body_pose_joint_val.double_value = 0.0
    # new sync
    def _sync_pose_joint_sliders(self):
        # sync joint sliders to represent actual values instead of resetting
        bm = self._body_model.selected_text
        bp = self._body_pose_comp.selected_text

        if not bp or self._body_pose_joint.number_of_items == 0 or \
                not self._body_pose_joint.selected_text:
            self._reset_rot_sliders()
            return

        ji = int(self._body_pose_joint.selected_text.split('-')[0])

        if bm == 'MHR':
            self._sync_mhr_value_slider(ji)
            self._body_pose_joint_x.int_value = 0
            self._body_pose_joint_y.int_value = 0
            self._body_pose_joint_z.int_value = 0
            return

        rotvec = AppWindow.POSE_PARAMS[bm][bp][0, ji].numpy()
        euler = rotvec_to_euler_for_body_model(bm, rotvec)
        self._body_pose_joint_x.int_value = int(round(euler[0]))
        self._body_pose_joint_y.int_value = int(round(euler[1]))
        self._body_pose_joint_z.int_value = int(round(euler[2]))
        self._body_pose_joint_val.double_value = 0.0

    def _sync_global_sliders(self):
        # same here translation and orientation sync instead of reset
        bm = self._body_model.selected_text
        params = AppWindow.POSE_PARAMS.get(bm, {})

        if "trans" in params:
            t = params["trans"][0, 0]
            self._trans_x.double_value = float(t[0])
            self._trans_y.double_value = float(t[1])
            self._trans_z.double_value = float(t[2])
        else:
            self._trans_x.double_value = 0.0
            self._trans_y.double_value = 0.0
            self._trans_z.double_value = 0.0

        if "global_orient" in params:
            r = params["global_orient"][0, 0]
            self._rot_x.double_value = float(r[0])
            self._rot_y.double_value = float(r[1])
            self._rot_z.double_value = float(r[2])
        elif "pose" in params:
            p = params["pose"][0, 0]
            if bm == 'ANNY':
                self._rot_x.double_value = float(p[0])
                self._rot_y.double_value = float(p[2])
                self._rot_z.double_value = float(-p[1])
            else:
                self._rot_x.double_value = float(p[0])
                self._rot_y.double_value = float(p[1])
                self._rot_z.double_value = float(p[2])
        else:
            self._rot_x.double_value = 0.0
            self._rot_y.double_value = 0.0
            self._rot_z.double_value = 0.0

    def _get_or_create_beta_tensor(self, name):
        if name not in AppWindow.BODY_BETAS:
            AppWindow.BODY_BETAS[name] = torch.zeros(1, AppWindow.BODY_MODEL_N_BETAS[name])
        return AppWindow.BODY_BETAS[name]

    def _get_or_create_exp_tensor(self, name):
        if name not in AppWindow.BODY_EXPS:
            AppWindow.BODY_EXPS[name] = torch.zeros(1, 10)
        return AppWindow.BODY_EXPS[name]

    def _sync_beta_slider(self):
        if self._body_model.selected_text == 'ANNY':
            name = self._body_model_shape_comp.selected_text
            self._body_beta_val.double_value = self._anny_phenotype_values.get(name, 0.5)
        else:
            idx = self._body_model_shape_comp.selected_index
            if idx < 0:
                idx = 0
            self._body_beta_val.double_value = self._body_beta_tensor[0, idx].item()

    def _sync_exp_slider(self):
        idx = self._body_model_exp_comp.selected_index
        if idx < 0:
            idx = 0
        self._body_exp_val.double_value = self._body_exp_tensor[0, idx].item()

    def _apply_transparency(self, val):
        if self._scene.scene.has_geometry("__body_model__"):
            mat_body = rendering.MaterialRecord()
            mat_body.shader = "defaultLitTransparency"
            mat_body.base_color = [0.9, 0.9, 0.9, 1 - val]
            self._scene.scene.modify_geometry_material("__body_model__", mat_body)

    def _sync_transparency(self):
        bm = self._body_model.selected_text
        val = AppWindow.TRANSPARENCY.get(bm, 0.0)
        self._transparency.double_value = val
        self._apply_transparency(val)

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_save_dialog(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.set_on_cancel(self._on_save_dialog_cancel)
        dlg.set_on_done(self._on_save_dialog_done)
        self.window.show_dialog(dlg)

    def _on_save_dialog_cancel(self):
        self.window.close_dialog()

    def _on_save_dialog_done(self, filename):
        self.window.close_dialog()
        output_dict = {
            'betas': self._body_beta_tensor,
            'expression': self._body_exp_tensor,
            'gender': self._body_model_gender.selected_text,
            'body_model': self._body_model.selected_text,
            'joints': AppWindow.JOINTS,
        }
        output_dict.update(AppWindow.POSE_PARAMS[self._body_model.selected_text])
        logger.debug(f'Saving output to {filename}')
        joblib.dump(output_dict, filename)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)
        self.window.set_needs_layout()

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Body Model Visualizer - Help"))
        dlg_layout.add_child(gui.Label("Select / drag: enable Drag joints on the toolbar, or hold Q"))
        dlg_layout.add_child(gui.Label("Release the mouse or Q to solve IK"))
        dlg_layout.add_child(gui.Label("-- Move selected Joint --"))
        dlg_layout.add_child(gui.Label("Move -x/+x: 1/2"))
        dlg_layout.add_child(gui.Label("Move -y/+y: 3/4"))
        dlg_layout.add_child(gui.Label("Move -z/+z: 5/6"))
        dlg_layout.add_child(gui.Label("After key adjustments, click Run IK in Model > Pose"))
        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def add_ground_plane(self):
        logger.info('drawing ground plane')
        gp = get_checkerboard_plane(plane_width=2, num_boxes=9)

        for idx, g in enumerate(gp):
            g.compute_vertex_normals()
            self._scene.scene.add_geometry(f"__ground_{idx:04d}__", g, self.settings._materials[Settings.LIT])

    # ANNY nutzt Phenotypes (Dict), andere Modelle Betas (Tensor)
    # damit wrapper gut funktioniert hilfsmethode
    def get_shape_params(self):
        bm = self._body_model.selected_text
        if bm == "ANNY":
            return self._anny_phenotype_values
        else:
            return self._body_beta_tensor


    # Restrict the models offered (e.g. a trimmed app bundle); None means all.
    REQUESTED_MODELS = None

    def preload_body_models(self):
        """Load every requested model variant whose wrapper and files exist."""
        requested = AppWindow.REQUESTED_MODELS
        for body_model in list(AppWindow.BODY_MODEL_NAMES):
            if requested is not None and body_model not in requested:
                AppWindow.BODY_MODEL_NAMES.remove(body_model)
                continue
            loaded_genders = []
            for gender in AppWindow.BODY_MODEL_GENDERS[body_model]:
                logger.info(f'Loading {body_model}-{gender}')
                try:
                    model = WRAPPER_CLASSES[body_model]().preload_body_model(gender)
                except ImportError as error:
                    # The model's package is not installed (or is broken).
                    logger.warning(f'{body_model} unavailable, skipping: {error}')
                    break
                except (OSError, AssertionError, ValueError, KeyError, RuntimeError) as error:
                    # Missing or unreadable model files: offer the other variants.
                    logger.warning(f'Skipping {body_model}-{gender}: {error}')
                    continue
                AppWindow.JOINT_NAMES[body_model] = model.parameter_names
                AppWindow.KEYPOINT_NAMES[body_model] = model.joint_names
                key = f'{body_model.lower()}-{gender.lower()}'
                AppWindow.PRELOADED_BODY_MODELS[key] = model
                loaded_genders.append(gender)
            if loaded_genders:
                AppWindow.BODY_MODEL_GENDERS[body_model] = loaded_genders
            else:
                logger.warning(f'No model files found for {body_model}; removing it')
                AppWindow.BODY_MODEL_NAMES.remove(body_model)
        if not AppWindow.BODY_MODEL_NAMES:
            raise RuntimeError('No body models could be loaded. Place model files '
                               f'under {external_models_dir()} (see README).')
        logger.info(f'Loaded body models {list(AppWindow.PRELOADED_BODY_MODELS)}')

    def load_body_model(self, body_model=None, gender=None):
        body_model = body_model or self._body_model.selected_text
        gender = gender or self._body_model_gender.selected_text
        model = AppWindow.PRELOADED_BODY_MODELS[f'{body_model.lower()}-{gender.lower()}']
        mesh_data = model.evaluate(
            AppWindow.POSE_PARAMS[body_model], self.get_shape_params(),
            expression=self._body_exp_tensor,
        )
        verts, joints, faces = mesh_data
        # Viewer offsets must not mutate arrays owned by the model.
        AppWindow.JOINTS = joints.copy()

        # bauen des 3D-Mesh
        mesh = o3d.geometry.TriangleMesh()

        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.5, 0.5, 0.5])

        # Compute the ground offset once per shape.
        if AppWindow.GROUND_OFFSET is None:
            user_y = 0.0
            if "trans" in AppWindow.POSE_PARAMS[body_model]:
                user_y = AppWindow.POSE_PARAMS[body_model]["trans"][0, 0, 1].item()
            base_min_y = mesh.get_min_bound()[1] - user_y
            AppWindow.GROUND_OFFSET = -base_min_y
        ground_offset = AppWindow.GROUND_OFFSET
        mesh.translate([0, ground_offset, 0])
        AppWindow.JOINTS += np.array([0, ground_offset, 0])
        self._ik_target_dirty = False
        AppWindow.IS_DRAGGING = False
        AppWindow.DRAG_DEPTH = None

        if self._scene.scene.has_geometry("__body_model__"):
            self._scene.scene.remove_geometry("__body_model__")
        self._scene.scene.add_geometry("__body_model__", mesh,
                                        self.settings.material)
        bounds = mesh.get_axis_aligned_bounding_box()
        if AppWindow.CAM_FIRST:
            self._scene.setup_camera(60, bounds, bounds.get_center())
            AppWindow.CAM_FIRST = False
            AppWindow.CAM_BOUNDS = bounds
            AppWindow.CAM_CENTER = bounds.get_center()

        AppWindow.BODY_TRANSL = torch.tensor([[0, ground_offset, 0]])
        self._on_show_joints(self._show_joints.checked, update_transparency=False)

        # add_geometry() above always attaches the default (opaque) material,
        # which would silently discard any transparency previously set for
        # this model every time the mesh is rebuilt (e.g. on every pose/beta
        # slider tweak, not just on model switch). Re-apply it here so the
        # rendered mesh always matches the transparency slider.
        self._apply_transparency(AppWindow.TRANSPARENCY.get(body_model, 0.0))

    def load(self, path):

        geometry = None
        geometry_type = o3d.io.read_file_geometry_type(path)

        mesh = None
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            mesh = o3d.io.read_triangle_mesh(path)
        if mesh is not None:
            if len(mesh.triangles) == 0:
                print(
                    "[WARNING] Contains 0 triangles, will read as point cloud")
                mesh = None
            else:
                mesh.compute_vertex_normals()
                if len(mesh.vertex_colors) == 0:
                    mesh.paint_uniform_color([1, 1, 1])
                geometry = mesh
            # Make sure the mesh has texture coordinates
            if not mesh.has_triangle_uvs():
                uv = np.array([[0.0, 0.0]] * (3 * len(mesh.triangles)))
                mesh.triangle_uvs = o3d.utility.Vector2dVector(uv)
        else:
            print("[Info]", path, "appears to be a point cloud")

        if geometry is None:
            cloud = None
            try:
                cloud = o3d.io.read_point_cloud(path)
            except Exception:
                pass
            if cloud is not None:
                print("[Info] Successfully read", path)
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                geometry = cloud
            else:
                print("[WARNING] Failed to read points", path)

        if geometry is not None:
            try:
                self._scene.scene.add_geometry("__model__", geometry,
                                               self.settings.material)
                bounds = geometry.get_axis_aligned_bounding_box()
                self._scene.setup_camera(60, bounds, bounds.get_center())
            except Exception as e:
                print(e)

    def export_image(self, path, width, height):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)


def main(args):
    if args.models:
        AppWindow.REQUESTED_MODELS = [name.strip().upper() for name in args.models.split(',')]
    if args.web:
        logger.info('Initializing web visualization')
        o3d.visualization.webrtc_server.enable_webrtc()

    # We need to initalize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = AppWindow(1920, 1080)

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--web', action='store_true', help='Enable web visualization')
    parser.add_argument('--models', default=os.environ.get('BMVIS_MODELS', ''),
                        help='Comma-separated subset of body models to load, e.g. SMPL,MANO '
                             '(default: every model whose files are present)')

    # macOS may pass a process-serial argument to a bundled app; ignore unknowns.
    args, _ = parser.parse_known_args()
    main(args)
