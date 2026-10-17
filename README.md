# Body Model Visualizer

## Introduction

Body Model Visualizer is an Open3D-based GUI for interactively exploring
parametric human body and part models. Users can modify shape, expression,
pose, and orientation parameters and inspect the resulting mesh immediately.


Main features include:

- Interactive editing of model parameters
- Body-pose and global-orientation controls
- Joint and joint-name visualization
- A simple IK solver for matching an input pose
- Export of edited model parameters
- View, lighting, transparency, and material controls
- Web visualization support

It was originally developed for the SMPL-family body models : SMPL, SMPL-X, MANO and FLAME. This repository extends the original workflow by integrating additional models through wrappers, while keeping the existing UI and rendering loop.
 
Added/integrated backends:

- SUPR, integrated directly into the existing application flow.`Wrapper/SUPR_WRAPPER.py` is therefore retained as a placeholder.
- STAR, integrated through a model-specific wrapper
- ANNY, integrated through a model-specific wrapper
- MHR, integrated through a model-specific wrapper

The following videos demonstrate the original UI features. They do not necessarily show the additional STAR, ANNY, MHR, or SUPR integrations.

- Interactive editing of shape, expression, pose parameters


https://user-images.githubusercontent.com/6137870/147476574-983063a8-233b-400c-bd64-7d946578919b.mp4


- Visualize body model joints and joint names


https://user-images.githubusercontent.com/6137870/147476577-39cd3a59-1add-4e2d-8c87-406ef964b558.mp4


- Simple IK solver to match an input pose


https://user-images.githubusercontent.com/6137870/147476585-9bbc0018-9220-4efa-9f4f-f37fdcf35db9.mp4


- Save edited model parameters


https://user-images.githubusercontent.com/6137870/147476590-d1b3e275-207e-4b30-99d6-0386f5ab74c5.mp4


- View controls


https://user-images.githubusercontent.com/6137870/147476594-cf244338-c841-4f17-a221-98038fdd9f4a.mp4


- Lighting controls


https://user-images.githubusercontent.com/6137870/147476612-ccd73006-4e7d-4caf-ae99-50418444f1fa.mp4


- Material settings


https://user-images.githubusercontent.com/6137870/147476625-8d019582-2a15-41f8-ae7f-93435e7e2529.mp4


- Web visualization support

Even though there are existing Blender/Unity plugins for these models, our main
audience here is researchers who would like to quickly edit/visualize body models
without the need to install a graphics software.


## Installation

Use the provided conda environment file (recommended):

```bash
conda env create -f environment.yml
```

Download the SMPL, SMPL-X, MANO, FLAME body models:

- SMPL: https://smpl.is.tue.mpg.de/ (v1.1.0)
- SMPL-X: https://smpl-x.is.tue.mpg.de/ (v1.1)
- MANO: https://mano.is.tue.mpg.de/
- FLAME: https://flame.is.tue.mpg.de/
  - For landmarks: https://github.com/soubhiksanyal/RingNet/blob/master/flame_model/
Additional models integrated in this repo:

- STAR: https://star.is.tue.mpg.de/
  - place STAR files under `data/body_models/star/` as `star_male.npz`, `star_female.npz`, `star_neutral.npz`
- SUPR: https://supr.is.tue.mpg.de/
  - place SUPR files under `data/body_models/supr/` as `SUPR_MALE.npy`, `SUPR_FEMALE.npy`, `SUPR_NEUTRAL.npy`
- ANNY source: https://github.com/naver/anny (included through submodule path `anny/`)
  - ANNY runtime data is read from `anny/src/anny/data` (not from `data/body_models/anny`)
- MHR source: https://github.com/facebookresearch/MHR
  - this repo already includes `mhr/assets/` expected by the wrapper

Copy downloaded files under `data/body_models` so the tree includes:

```text
data
└── body_models
    ├── flame
    │   ├── FLAME_FEMALE.pkl
    │   ├── FLAME_MALE.pkl
    │   ├── FLAME_NEUTRAL.pkl
    │   ├── flame_dynamic_embedding.npy
    │   └── flame_static_embedding.pkl
    ├── mano
    │   ├── MANO_LEFT.pkl
    │   └── MANO_RIGHT.pkl
    ├── smpl
    │   ├── SMPL_FEMALE.pkl
    │   ├── SMPL_MALE.pkl
    │   └── SMPL_NEUTRAL.pkl
    ├── star
    │   ├── star_female.npz
    │   ├── star_male.npz
    │   └── star_neutral.npz
    ├── supr
    │   ├── SUPR_FEMALE.npy
    │   ├── SUPR_MALE.npy
    │   └── SUPR_NEUTRAL.npy
    └── smplx
        ├── SMPLX_FEMALE.npz
        ├── SMPLX_MALE.npz
        └── SMPLX_NEUTRAL.npz
```

Note for case-sensitive filesystems (Linux): SUPR loader in `main.py` currently uses lower-case names like `supr_male.npy`.
If your filesystem is case-sensitive, keep filenames consistent with loader expectation.

ANNY data used by `Wrapper/ANNY_WRAPPER.py` is loaded from the submodule path `anny/src/anny/data/`. Typical structure is:

```text
data
└── body_models
 └──anny/
  └── src
   └── anny
    └── data
      ├── mpfb2/
      ├── shape_calibration/
      └── ...
```

MHR assets used by `Wrapper/MHR_WRAPPER.py` are loaded from `mhr/assets/`. Typical files are:

```text
data
└── body_models
 └──mhr/
  └── assets
    ├── compact_v6_1.model
    ├── corrective_activation.npz
    ├── corrective_blendshapes_lod0.npz ... corrective_blendshapes_lod6.npz
    ├── lod0.fbx ... lod6.fbx
    └── mhr_model.pt
```
  


Finally, run:
```shell
python main.py
```
If you want to enable web visualization, run:
```shell
python main.py --web
```

## Guidelines

### Saved model parameters
`File > Save Model Params` lets you save the edited body model parameters. Output is a pickled
python dictionary with below keys:
```shell
dict_keys(['betas', 'expression', 'gender', 'body_model', 
           'joints', 'body_pose', 'global_orient'])
```

### Model interface and checks

All eight model types use `BodyModelWrapper`. `evaluate()` disables gradient
tracking and returns `MeshData(vertices, joints, faces)` as NumPy arrays. Wrappers
expose `joint_names`, `joint_parents` (root parent is `-1`), `parameter_names`,
and `parameter_limits`. Parameter metadata is keyed by pose group; an absent
limit group uses the viewer's rotation controls. Extra surface landmarks are
not part of the kinematic tree. `ik_joints()` returns differentiable positions
in viewer coordinates, and `ik_parameter_masks()` identifies editable pose
parameters for the shared solver in `inverse_kinematics.py`.

### Inverse kinematics

IK supports SMPL, SMPL-X, MANO, FLAME, SUPR, STAR, ANNY, and MHR:

1. Enable **Drag joints** on the viewport toolbar; this also shows the skeleton.
2. Click and drag a joint marker, then release to fit the pose. On a trackpad,
   press and hold with one finger while moving it. Turn off Drag joints to orbit
   the camera. Holding **Q** while dragging is also supported when the viewport has keyboard focus.
3. Alternatively, select a joint, use **1/2**, **3/4**, or **5/6** to adjust its
   target along X, Y, or Z, and click **Run IK** in the Pose section.

The sidebar has two tabs. **Model** holds model selection plus collapsible
Shape, Expression, Pose, and Global transform sections; **Display** holds
transparency, lighting, and material controls. Every parameter slider has a
number field next to it for exact values. The toolbar over the viewport
offers Reset view, Front/Side/Top camera presets, and the Skeleton, Labels,
and Drag joints toggles.

IK optimizes joint/landmark positions while holding body shape, facial
expression, root rotation, and translation fixed. Other points act as soft
anchors, so nearby joints can move to reach a target. MHR's native pose limits
are enforced and its scale/bone-length controls are excluded. Other models use
their existing rotational parameterizations and a penalty for changing the
current pose, without additional anatomical joint-limit data. FLAME fits head,
jaw, and eye articulation to its joint/landmark targets; it does not generate
new facial expressions. A fixed root or an unreachable target may retain some
position error, which is shown in the status message.

Run the regression tests in the configured application environment:

```shell
python -m unittest discover -s tests -v
```

To exercise model switching, shape and pose controls, skeletons, joint labels,
and IK in an automatically closed GUI window (requires local model assets and
a desktop session):

```shell
python tests/gui_smoke.py
```

Run numerical IK checks against all eight models (requires local assets):

```shell
python tests/ik_models.py
```
