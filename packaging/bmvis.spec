# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for a trimmed, self-contained Body Model Visualizer.

Build from the repository root on the target OS (PyInstaller cannot
cross-compile); BUNDLE only has an effect on macOS, elsewhere the result is
the folder dist/<APP_NAME>/ with the executable inside:
    python -m PyInstaller --noconfirm packaging/bmvis.spec

Only the model files listed in BUNDLED_MODELS are copied in; the app discovers
what is present at start-up and offers just those models. SMPL-family files
are MPI-licensed and must not be redistributed, so keep such builds personal.
"""
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = os.path.abspath(os.path.join(os.path.dirname(SPEC), os.pardir))
APP_NAME = 'Body Model Visualizer'

# (source relative to the repository, destination folder inside the bundle)
BUNDLED_MODELS = [
    ('data/body_models/smpl/SMPL_NEUTRAL.pkl', 'data/body_models/smpl'),
    ('data/body_models/mano/MANO_LEFT.pkl', 'data/body_models/mano'),
    ('data/body_models/mano/MANO_RIGHT.pkl', 'data/body_models/mano'),
]

# Files that are not present are simply left out; users can add any model
# later by creating data/body_models next to the app (see paths.py).
datas = [(os.path.join(ROOT, src), dst) for src, dst in BUNDLED_MODELS
         if os.path.exists(os.path.join(ROOT, src))]
binaries, hiddenimports = [], []
for package in ('open3d', 'smplx', 'chumpy', 'roma'):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += collect_submodules('scipy.spatial.transform')
# Wrapper modules are imported by name at runtime (Wrapper/wrapper_dict.py).
hiddenimports += ['Wrapper', 'Wrapper.body_model_wrapper', 'Wrapper.SMPL_WRAPPER', 'paths']

# Only exclude what is certainly unrelated: the other body-model packages
# (their wrappers import lazily and are skipped when absent) and GUI toolkits.
# Open3D imports plotly, matplotlib, dash, pandas and scikit-learn at module
# level, so general-purpose packages are left for PyInstaller to resolve.
excludes = [
    'anny', 'mhr', 'pymomentum', 'warp', 'STAR', 'SUPR',
    'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'jupyter', 'notebook',
]

a = Analysis(
    [os.path.join(ROOT, 'main.py')],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    strip=False,
    upx=False,
    console=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=APP_NAME)
app = BUNDLE(
    coll,
    name=f'{APP_NAME}.app',
    icon=None,
    bundle_identifier='org.bmvisualizer.app',
    info_plist={
        'NSHighResolutionCapable': True,
        'CFBundleShortVersionString': '0.1.0',
        'LSMinimumSystemVersion': '12.0',
    },
)
